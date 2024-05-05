from lib import *
num_cols = joblib.load('file/num_cols.pkl')
scaler = joblib.load('file/scaler.pkl')

def preprocessing(X):
    X_tf = scaler.transform(X)

    return X_tf



def load_model(net, model_path):
    load_weight = torch.load(model_path)
    net.load_state_dict(load_weight)
    print(net)
    # return net
