from .feedforward import *


class RecurrentCoach(FeedForwardCoach):
    """
    Class which facilitates training of recurrent search spaces or models, based on such spaces,
    given they follow interface defined in scripts.GenericModel.

    Args:
        space (nn.Module): SearchSpace or GenericModel and their descendants
        sequence_loss (bool): wheter to calculate loss based on output sequence rather than the last element of it
        **kwargs (dict): keyword arguments passed to FeedForwardCoach constructor
    """
    def __init__(self, space, sequence_loss=False, **kwargs):
        super().__init__(space, **kwargs)

        self.sequence_loss = sequence_loss

    def train(self, epochs=1, steps=None, preserve_states=False, *args, **kwargs):
        """
        Train for ``epochs`` epochs, each of ``steps`` batches.

        Args:
            epochs (int): number of epochs to train
            steps (int, optional): number of batches in one epoch
            preserve_states (bool): whether to preserve states across batches
        """
        super().train(epochs, steps, preserve_states=preserve_states, *args, **kwargs)

    def _training_loop_body(self, input, labels, description, preserve_states=True,
                            progress_bar=None, *args, **kwargs):
        """
        A training loop, which wraps ``steps`` calls to ``_training_loop_body``.
        If ``steps`` is not specified, defaults to  ``len(loaders.train)``.
        """
        if not preserve_states: self.states = None

        loss, self.states = self.get_loss(description, input, labels,
                                          self.states, *args, **kwargs)

        self.optim.zero_grad()
        loss.backward()

        if self.grad_norm:
            clip_grad_norm(self.space.parameters(), self.grad_norm)

        self.optim.step()

        self.stats.losses.append(loss.item())
        if progress_bar is not None:
            progress_bar.set_postfix_str(f'Loss: {loss.item():5.3f}')

    def evaluate(self, description, loader=None, preserve_states=False, *args, **kwargs):
        """
        Evaluates trained model on data from ``loader`` or
        ``self.loaders.validation`` if the former is not specified.

        Args:
            description (dict): description to evaluate.
            loader (DataLoader, optional): loader with validation data
            preserve_states (bool): whether to preserve states across batches

        Returns:
            Model performance stats collected during evaluation
        """
        self.clear_stats()
        loader = loader or self.loaders.validation
        if self.model is None:
            raise ValueError('You must train the model first.')
        self.model = self.model.eval()

        progress_bar = self.tqdm(loader, desc=f'Evaluating {self.name} on {str(self.space.device)}')
        for i, (input, labels) in enumerate(progress_bar):
            with torch.no_grad():
                if not preserve_states: self.states = None
                loss, self.states = self.get_loss(description, input, labels,
                                                  hidden=self.states, *args, **kwargs)
                self.stats.losses.append(loss.item())

                progress_bar.set_postfix_str(f'Loss: {loss.item():5.3f}')

        if self.logger is not None:
            self.logger.info(f'{self.name} evaluation loss is: '
                             f'{np.mean(self.stats.losses):5.3f}')
        return self.stats

    def get_loss(self, description, inputs, labels, hidden=None, mean=True):
        """
        A method which handles calculating loss and all that comes with it.

        Args:
             description (dict): description of the graph to evaluate
             inputs (Tensor): pretty self-explanatory
             labels (Tensor): labels, corresponding to inputs
             hidden (list, tuple): hidden states passed to the recurrent cell
             mean (bool): whether or not to return mean value of losses.

        Returns:
            loss.
        """
        inputs = wrap(inputs, self.space.device)
        labels = wrap(labels, self.space.device)

        if not hasattr(self, 'model'):
            self.model = self.space.prepare(description)
        states = self.model(inputs=inputs, description=description,
                            h=hidden, return_sequence=self.sequence_loss)
        out = states[0]

        loss = self.criterion(out, labels)

        if 'auc' in self.metrics:
            out = F.sigmoid(out.detach())
            try:
                self.stats.auc.append(roc_auc_score(get_np(labels), get_np(out)))
            except ValueError:
                pass

        return loss.mean() if mean else loss, states