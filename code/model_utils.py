import keras.backend as K


def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_and_vec(x):
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def seq_maxpool(x):    
    seq, mask = x
    mask = K.expand_dims(mask, axis=-1)
    seq -= (1 - mask) * 1e10  ## MAX_NUMBER when fp32/fp16
    return K.max(seq, 1, keepdims=True)

