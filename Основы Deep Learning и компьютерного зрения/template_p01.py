import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    #формула
    scores = decoder_hidden_state.T @ W_mult @ encoder_hidden_states
    # нормализуем, чтобы свести к единице
    weights = softmax(scores) 
    #умножаем
    attention_vector = weights @ encoder_hidden_states.T  
    return attention_vector.T  

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    #первое умножение 
    enc_term = W_add_enc @ encoder_hidden_states 
    #второе умножение 
    dec_term = W_add_dec @ decoder_hidden_state
    #нужна одинаковая форма чтобы их сложить
    dec_term_broadcasted = np.tile(dec_term, (1, encoder_hidden_states.shape[1]))  
    combined = enc_term + dec_term_broadcasted
    #функция активации
    tanh_combined = np.tanh(combined)
    #умножаем на v^t
    scores = v_add.T @ tanh_combined
    #как и в прошлый раз
    weights = softmax(scores)                    
    attention_vector = weights @ encoder_hidden_states.T  
    return attention_vector.T                     