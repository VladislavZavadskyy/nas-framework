Configuring the search
======================

In order to configure a search process, YAML configuration file must be created.
The configuration file should have the following structure:

.. code-block:: yaml

    log_dir: ...
    data_dir: ...

    searchspace:
      ...

    child_training:
      ...

    architect_training:
      ...

Search space section
^^^^^^^^^^^^^^^^^^^^

Search space section should contain search space type name:

.. code-block:: yaml

    searchspace:
      type: ... # RNN or MLP

And keyword arguments to be passed to search space constructor of the specified type.

Child training section
^^^^^^^^^^^^^^^^^^^^^^

Child training section should contain the following:

batch_size
~~~~~~~~~~

The size of batch of data to be processed by child network each training step.

.. code-block:: yaml

    child_training:
      batch_size: ...

adaptive_batch_size
~~~~~~~~~~~~~~~~~~~

Boolean, which specifies whether adaptive batch size should be used.

.. note::

    Adaptive batch size decreases batch size by multiplying it by ``batch_size_decay`` each time when
    CUDA OOM error is encountered until ``min_batch_size`` is reached. Using this, allows to evaluate
    memory-demanding architectures, instead of skipping them.

.. code-block:: yaml

    child_training:
      adaptive_batch_size: ...


min_batch_size
~~~~~~~~~~~~~~

Minimal batch size, after reaching which, attempts to evaluate an architecture will be abandoned.
Optional if ``adaptive_batch_size`` is ``false``.

.. code-block:: yaml

    child_training:
      min_batch_size: ...

max_batch_size
~~~~~~~~~~~~~~

Maximal batch size, which is also the initial one if ``adaptive_batch_size`` is enabled.
Optional if ``adaptive_batch_size`` is ``false``.

.. code-block:: yaml

    child_training:
      max_batch_size: ...

batch_size_decay
~~~~~~~~~~~~~~~~

Float, by which current batch size is multiplied if OOM is encountered.
Should be in range :math:`(0,1)`.
Optional if ``adaptive_batch_size`` is ``false``.

.. code-block:: yaml

    child_training:
      batch_size_decay: ...


keep_data_on_device
~~~~~~~~~~~~~~~~~~~

Boolean, which indicates whether the dataset should be stored in device memory entirely.

.. code-block:: yaml

    child_training:
      keep_data_on_device: ...

.. warning::

    Enabling this option may increase device memory usage drastically.

.. note::
    Any other specified keyword argument will be passed to the ``FeedForwardCoach`` constructor.

Architect training section
^^^^^^^^^^^^^^^^^^^^^^^^^^

curriculum
~~~~~~~~~~

Boolean, which indicates whether a curriculum learning procedure should be used.

.. code-block:: yaml

    architect_training:
        curriculum: ...

max_curriculum_complexity
~~~~~~~~~~~~~~~~~~~~~~~~~

Maximal complexity to be used during the curriculum training procedure.
When this complexity is reached, a normal training procedure begins with sampling data from all
data, generated during the curriculum training procedure.

Optional if ``curriculum`` is ``false``.

.. code-block:: yaml

    architect_training:
        max_curriculum_complexity: ...


epochs_per_loop
~~~~~~~~~~~~~~~

Number of architect training epochs per sample-evaluate-train loop.

.. code-block:: yaml

    architect_training:
        epochs_per_loop: ...

lr_decay
~~~~~~~~

Facrtor by which architect learning rate is multiplied each loop.
Should be in range :math:`(0,1)`.

.. code-block:: yaml

    architect_training:
        lr_decay: ...

storage_surplus_factor
~~~~~~~~~~~~~~~~~~~~~~

The factor by which number of stored data points (description-rewards)
must surpass minimum required at the current curriculum level.
Should be in range :math:`[1,\infty)`.

.. code-block:: yaml

    architect_training:
        storage_surplus_factor: ...

load_architect
~~~~~~~~~~~~~~

Boolean, which specifies whether a checkpoint of architect model should be loaded, if it exists.

.. code-block:: yaml

    architect_training:
        load_architect: ...

.. note::
    Any other specified keyword argument will be passed to the ``Architect`` constructor.