from lib import *
from utils import load_model

class_index = ['low_income', 'high_income']

class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, output):
        index = np.argmax(output.detach().numpy())
        predict_label = self.class_index[index]

        return predict_label


predictor = Predictor(class_index)

def predict(X):
    # Prepare network
    model_path = 'file/weights.pth'

    net.eval()

    # Prepare model
    model = load_model(net, model_path)
    output = model(X)
    response = predictor.predict_max(output)

    return response


X = np.array([[0.2329, 0.2233, 0.2667, 0.0000, 0.0000, 0.3980, 0.0000, 0.0000, 0.0000,
               1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
               1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
               0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]])
X = torch.Tensor(X)

result = predict(X)
print(result)
