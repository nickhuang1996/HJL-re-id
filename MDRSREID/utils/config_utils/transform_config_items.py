
def transfer_config_items(src, dest):
    """
    :param src: cfg.dataset or cfg.dataloader
    :param dest: cfg.dataset or cfg.dataloader
    :return: cfg.dataset or cfg.dataloader(has been changed)

    Get the src attribute.Attention: src and dest are the same always.
    The attribute may have many key:values pairs, like
    Attribute A: *
    Attribute B: *
    Attribute C: *
    ......
    Though this function we just put these keys outside the selected attribute and they become the new attributes.
    """
    for k, v in src.items():
        dest[k] = src[k]
