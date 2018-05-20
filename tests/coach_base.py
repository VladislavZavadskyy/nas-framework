from unittest import TestCase
from nasframe.coaches.base import CoachBase
from torch import nn


class TestCoachBase(TestCase):
    def test_update_convergence_status(self):
        num_steps = 10
        coach_base = CoachBase(nn.Linear(1,1),
                               convergence_patience=num_steps,
                               steps_per_epoch=100)

        for i in range(num_steps+1):
            self.assertFalse(coach_base.is_converged)
            coach_base.update_convergence_status(1)

        self.assertTrue(coach_base.is_converged)

        coach_base.convergence.mode = 'max'
        coach_base.reset()

        for i in range(num_steps+1):
            self.assertFalse(coach_base.is_converged)
            coach_base.update_convergence_status(1)

        self.assertTrue(coach_base.is_converged)