import torch
import pandas as pd
from keras_preprocessing.text import Tokenizer
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gib2kor
from dekorde.builders import build_X, build_Y, build_lookahead_mask
from dekorde.utils import seed_everything, load_model


def main():
    conf = load_conf()
    # --- conf; hyper parameters --- #
    device = load_device()
    seed = conf['seed']
    max_length = conf['max_length']
    hidden_size = conf['hidden_size']
    heads = conf['heads']
    num_layers = conf['num_layers']
    dropout = conf['dropout']
    lr = conf['lr']
    
    # --- build the data --- #
    seed_everything(seed)
    gib2kor = load_gib2kor(mode='train')
    gibs = [gib for gib, _ in gib2kor]
    kors = ["s" + kor for _, kor in gib2kor]  # s stands for "start of the sequence"
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    vocab_size = len(tokenizer.word_index.keys())
    X = build_X(gibs, tokenizer, max_length, device)  # (N, L)
    Y = build_Y(kors, tokenizer, max_length, device)  # (N, 2, L)
    lookahead_mask = build_lookahead_mask(max_length, device)  # (L, L)
    
    # --- load the model ---- #
    model = Transformer(device, hidden_size, vocab_size, max_length, heads, num_layers, dropout, lookahead_mask).to(device)
    model = load_model(model, 'model.pt')
    
    # --- do inference --- #
    model.eval()
    with torch.no_grad():
        max_indices = model.inference(X, Y).numpy() # convert to numpy for sequences_to_texts
    # Ref : https://github.com/keras-team/keras-preprocessing/blob/1.1.2/keras_preprocessing/text.py#L141-L487
    result = tokenizer.sequences_to_texts(max_indices) # 
    
    # --- save the result --- #
    strFormat = '%-40s%-40s%-40s\n'
    strOut = strFormat % ('|| source','|| target','|| result')
    
    for i, text in enumerate(result):
        strOut += strFormat %(gib2kor[i][0], gib2kor[i][1], text)
    print(strOut)
  
    result_df = pd.DataFrame.from_dict({'source': gibs, 'target': kors, 'result': result})
    result_df.to_csv("result.csv", index=False, encoding='utf-8-sig')
    
    
if __name__ == '__main__':
    main()
