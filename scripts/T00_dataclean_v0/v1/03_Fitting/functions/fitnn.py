from sklearn.model_selection import train_test_split
import torch 
import numpy as np

def fitnn(df, inputs_list, outputs_list,params):

    inputs_full  = df[inputs_list]
    outputs_full = df[outputs_list]

    # test 0.2

    inputs, inputs_test, outputs, outputs_test = train_test_split(inputs_full, outputs_full, test_size=0.15)

    # normalize 
    in_mu = inputs.mean()
    in_std = inputs.std()

    out_mu = outputs.mean()
    out_std = outputs.std()

    in_norm = (inputs - in_mu) / in_std
    out_norm = (outputs - out_mu) / out_std


    # convert to tensor
    in_norm = torch.tensor(in_norm.values, dtype=torch.float32)
    out_norm = torch.tensor(out_norm.values, dtype=torch.float32)

    hidden = params['hidden']
    neurons = params['neurons']

    class Net(torch.nn.Module):
        def __init__(self,hidden=hidden,neurons=neurons):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(in_norm.shape[1], neurons)
            self.fc2 = [ torch.nn.Linear(neurons, neurons) for i in range(hidden)]
            self.fc3 = torch.nn.Linear(neurons, out_norm.shape[1])
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            for i in range(hidden):
                x = torch.relu(self.fc2[i](x))
            x = self.fc3(x)
            return x

        def compute_l2_loss(self, w):
            return torch.pow(w, 2).sum()
    net = Net( hidden, neurons)


    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    for epoch in range(params['epochs']):

        optimizer.zero_grad()
        outputs_hat = net(in_norm)

        l2_weight = params['l2_weight']

        l2 = l2_weight * net.compute_l2_loss(net.fc1.weight)

        loss = criterion(outputs_hat, out_norm)  + l2
        loss.backward()
        optimizer.step()
        if epoch % 5000 == 0:
            loss_print = round(loss.item(), 4)
            print(f'epoch {epoch}, loss {loss_print}')

    predicted = net(in_norm).detach().numpy()
    predicted = predicted * out_std.values + out_mu.values

    # test 
    in_test = (inputs_test - in_mu) / in_std
    in_test = torch.tensor(in_test.values, dtype=torch.float32)

    out_test = (outputs_test - out_mu) / out_std

    out_test = torch.tensor(out_test.values, dtype=torch.float32)

    predicted_test = net(in_test).detach().numpy()

    predicted_test = predicted_test * out_std.values + out_mu.values



    # error train and test relative
    
    error_train = 100*np.abs(predicted - outputs.values) / outputs.values
    error_test = 100*np.abs(predicted_test - outputs_test.values) / outputs_test.values

    et_mean = error_train.mean()
    et_std = error_train.std()

    etest_mean = error_test.mean()
    etest_std = error_test.std()



    return {
        'predicted': predicted,
        'in_mu': in_mu,
        'in_std': in_std,
        'out_mu': out_mu,
        'out_std': out_std,
        'predicted_test': predicted_test,
        'inputs_test': inputs_test,
        'outputs_test': outputs_test,
        'predicted_train': predicted,
        'inputs_train': inputs,
        'outputs_train': outputs,
        "et_mean": et_mean,
        "et_std": et_std,
        "etest_mean": etest_mean,
        "etest_std": etest_std
    }