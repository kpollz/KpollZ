from lib import *
trial = joblib.load('file/trial.pkl')
inputs = 105
Path = 'file/weights.pth'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inputs,trial.params['n_units_l0'])
        self.dropout1 = nn.Dropout(trial.params['dropout_l0'])
        self.fc2 = nn.Linear(trial.params['n_units_l0'], trial.params['n_units_l1'])
        self.dropout2 = nn.Dropout(trial.params['dropout_l1'])
        self.fc3 = nn.Linear(trial.params['n_units_l1'], 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))

        return x

model = Model()
model.load_state_dict(torch.load(Path))
model.eval()
print(model)