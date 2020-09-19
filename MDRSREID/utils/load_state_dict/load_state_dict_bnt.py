def load_state_dict_bnt(model, src_state_dict, fold_bnt=True):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.

    Arguments:
        model: A torch.nn.Module object.
        src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
        This is modified from torch.nn.modules.module.load_state_dict(), to make
        the warnings and errors more detailed.
    """
    from torch.nn import Parameter

    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        # except Exception, msg:
        #     print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))
        except Exception as msg:
            print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))

    # New version of BN has buffer `num_batches_tracked`, which is not used
    # for normal BN, so we fold all these missing keys into one line
    def _fold_nbt(keys):
        nbt_keys = [s for s in keys if s.endswith('.num_batches_tracked')]
        if len(nbt_keys) > 0:
            keys = [s for s in keys if not s.endswith('.num_batches_tracked')] + ['num_batches_tracked  x{}'.format(len(nbt_keys))]
        return keys

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        if fold_bnt:
            src_missing = _fold_nbt(src_missing)
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        if fold_bnt:
            dest_missing = _fold_nbt(dest_missing)
        for n in dest_missing:
            print('\t', n)
