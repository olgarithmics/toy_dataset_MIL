from tensorflow.keras import callbacks as cbks

def fit_models(callback_model,
               models,
               generators,
               batch_size,
               steps_per_epoch=None,
               epochs=1,
               verbose=1,
               callbacks=None,
               initial_epoch=0):
    epoch = initial_epoch

    callback_metrics = sum([modelt.metrics_names for modelt in models], [])

    for model in models:
        model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=None)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=None))
    _callbacks += (callbacks or []) + [model.history for model in models]

    callbacks = _callbacks

    [callback.set_model(callback_model) for callback in callbacks]
    [callback.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': False,
        'metrics': callback_metrics,
    }) for callback in callbacks]

    [callback.on_train_begin() for callback in callbacks]

    try:
        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            [callback.on_epoch_begin(epoch) for callback in callbacks]
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:

                # build batch logs
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                [callback.on_batch_begin(batch_index, batch_logs) for callback in callbacks]

                for model, output_generator in zip(models, generators):
                    metrics = model.metrics_names

                    generator_output = next(output_generator)


                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 2:
                        x, y = generator_output

                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    outs = model.train_on_batch(x, y, sample_weight=sample_weight)

                    if not isinstance(outs, list):
                        outs = [outs]

                    for i, name in enumerate(metrics):
                        batch_logs[name] = outs[i]

                [callback.on_batch_end(batch_index, batch_logs) for callback in callbacks]

                batch_index += 1
                steps_done += 1

                if callback_model.stop_training:
                    break

            [callback.on_epoch_end(epoch, epoch_logs) for callback in callbacks]
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        pass

    [callback.on_train_end() for callback in callbacks]

    return [model.history for model in models]
