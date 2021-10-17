import torch
from torch.optim import Adam, AdamW
from keras_preprocessing.text import Tokenizer
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gib2kor
from dekorde.builders import build_X, build_Y, build_lookahead_mask
from dekorde.utils import seed_everything, save_checkpoint


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

    # --- instantiate the model and the optimizer --- #
    model = Transformer(device, hidden_size, vocab_size, max_length, heads, num_layers, dropout, lookahead_mask).to(device)
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # --- start training --- #
    model.train()
    for epoch in range(conf['epochs']):
        loss = model.training_step(X, Y)  # compute the loss
        loss.backward()  # backprop
        optimizer.step()  # gradient descent
        optimizer.zero_grad()  # prevent the gradients accumulating.
        print(f"epoch:{epoch}, loss:{loss}")

    # --- save the model --- #
    model_to_save = model.module if hasattr(model, 'module') else model
    save_checkpoint({
        'state_dict': model_to_save.state_dict()
    }, 'model.pt')

    
if __name__ == '__main__':
    main()