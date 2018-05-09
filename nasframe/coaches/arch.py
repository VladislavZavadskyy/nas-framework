from .base import CoachBase
from nasframe.utils.torch import *
from nasframe.utils.misc import selfless

from functools import partial
from numpy import random


class ArchitectCoach(CoachBase):
    """
    Class which facilitates architect network training with PPO algorithm.

    Args:
        architect (Architect): architect to be trained
        name (str): name of the architect, defaults to 'architect'
        storage (Storage, CurriculumStorage): storage of data required for the reinforcement learning of architect
        curriculum (bool): whether to train using curriculum training procedure
        epochs_to_increase (int, optional): epochs needed to pass for the level of curriculum to be increased
        random_complexity_chance (float): a chance of choosing a lower curriculum level for an epoch
        batch_size (int): number of descriptions to be processed per training step
        entropy_gain (float): factor by which mean entropy is multiplied before it's subtracted from total loss
        complexity_penalty (float): reward penalty for 1 million parameters
        ppo_clip (float): PPO clipping factor
        **kwargs: keyword arguments passed to CoachBase constructor

    """
    def __init__(self, architect, storage, name='architect',
                 curriculum=True, epochs_to_increase=10, lower_complexity_chance=.2,
                 batch_size=8, entropy_gain=1e-4, complexity_penalty=0.,
                 ppo_clip=.2, **kwargs):

        kwargs['scheduler_metric'] = kwargs.get('scheduler_metric', 'total_loss')
        super().__init__(model=architect, **kwargs)

        self.name = name

        self.entropy_gain = entropy_gain
        self.storage = storage
        self.batch_size = batch_size
        self.ppo_clip = ppo_clip
        self.complexity_penalty = complexity_penalty

        self.curriculum = curriculum
        self.epochs_to_increase = epochs_to_increase
        self.lower_complexity_chance = lower_complexity_chance
        self.stats.curriculum_complexity = 1*curriculum
        self.stats.epochs_until_increase = epochs_to_increase
        self._last_updated_epoch = 0

        self.metrics.update({
            'total_loss': {
                'fn': 'np_mean',
                'kwargs': ['total_losses']
            },
            'action_loss': {
                'fn': 'np_mean',
                'kwargs': ['action_losses']
            },
            'value_loss': {
                'fn': 'np_mean',
                'kwargs': ['value_losses']
            },
            'entropy': {
                'fn': 'np_mean',
                'kwargs': ['entropies']
            },
            'lr': {
                'fn': self.get_current_lr,
                'kwargs': []
            },
            'curriculum_complexity': {
                'fn': self.get_curriculum_complexity,
                'kwargs': []
            },
        })

    def get_curriculum_complexity(self):
        return self.curriculum_complexity

    @property
    def curriculum_complexity(self):
        if self.curriculum:
            if self.stats.epoch != self._last_updated_epoch:
                if self.stats.epochs_until_increase <= 0:
                    self.stats.epochs_until_increase = self.epochs_to_increase
                    self.stats.curriculum_complexity += 1
                else:
                    self.stats.epochs_until_increase -= 1
                self._last_updated_epoch = self.stats.epoch
            return self.stats.curriculum_complexity
        return 0

    def _training_loop(self, steps, explore=None, *args, **kwargs):
        if self.curriculum:
            current_complexity = self.curriculum_complexity
            if random.uniform() < self.lower_complexity_chance and current_complexity > 1:
                current_complexity = np.random.randint(1, current_complexity)
            self.storage.set_complexity(current_complexity)
        epoch_indices = random.choice(np.arange(len(self.storage)), self.batch_size*steps, replace=False)

        progress_bar = self.tqdm(range(steps), desc=f"Training {self.name} on {self.model.device}")
        for step in progress_bar:
            batch_indices = epoch_indices[step*self.batch_size:(step+1)*self.batch_size]
            self._training_loop_body(**selfless(locals()))

            if self.stats.accumulated_step % self.log_every == 0 and self.stats.accumulated_step > 0:
                self._log(**selfless(locals()))

            self.stats.accumulated_step += 1
            self.stats.step += 1

    def _training_loop_body(self, batch_indices, progress_bar, *args, **kwargs):
        action_losses, values_losses, mean_entropies = [], [], []

        for description, logps_old, values_old, entropies_old, \
                rewards, adv_targets, complexity in self.storage[batch_indices]:

            _, logps_new, values_new, entropies_new = self.model.evaluate_description(description)

            if self.complexity_penalty is not None and self.complexity_penalty > 0:
                adv_targets = adv_targets - complexity*self.complexity_penalty
                rewards = rewards - complexity*self.complexity_penalty

            ratio = torch.exp(logps_new - logps_old)
            surrogate = ratio * adv_targets
            surrogate_pess = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv_targets

            action_losses.append(-torch.min(surrogate, surrogate_pess).mean())
            values_losses.append((rewards - values_new).pow(2).mean())
            mean_entropies.append(entropies_new.view(-1))

            self.storage.update(description, logps_new, values_new, entropies_new)

        mean_action_loss = torch.stack(action_losses).mean()
        mean_value_loss = torch.stack(values_losses).mean()
        mean_entropy = torch.cat(mean_entropies).mean()
        total_loss = mean_action_loss + mean_value_loss - mean_entropy * self.entropy_gain

        self.optim.zero_grad()
        total_loss.backward()
        if self.grad_norm is not None and self.grad_norm > 0:
            nn.utils.clip_grad_norm(self.model.parameters(), self.grad_norm)
        self.optim.step()

        self.stats.entropies.append(mean_entropy.item())
        self.stats.value_losses.append(mean_value_loss.item())
        self.stats.action_losses.append(mean_action_loss.item())
        self.stats.total_losses.append(total_loss.item())

        progress_bar.set_postfix_str(
            f"Action loss: {self.stats.action_losses[-1]:6.4f} | "
            f"Value loss: {self.stats.value_losses[-1]:6.4f} | "
            f"Total loss: {self.stats.total_losses[-1]:6.4f} |"
            f"Entropy: {self.stats.entropies[-1]:6.4f}"

        )

    def _init_epoch(self, *args, **kwargs):
        super()._init_epoch(*args, **kwargs)
        self.stats.value_losses = []
        self.stats.action_losses = []
        self.stats.total_losses = []
        self.stats.entropies = []

    def train(self, epochs=1, steps=None, explore=True, *args, **kwargs):
        """
        Train architect for `epochs` epochs.
        """
        steps = steps or self.epoch_steps
        if steps*self.batch_size > len(self.storage):
            raise ValueError("Storage does not contain enough data points.")
        for epoch in range(epochs):
            self._init_epoch()
            self._training_loop(steps, explore)
            self._finalize_epoch()
