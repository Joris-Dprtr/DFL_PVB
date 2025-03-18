import torch
import time
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt


def save_model(model, name):
    """
    Saves the state dictionary using torch
    :param name: name of the file
    :param model: the model for which we want to save the state dictionary
    """
    torch.save(model.state_dict(), '../models/' + str(name))


class Training:

    def __init__(
            self,
            model,
            cvx_layer,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            T,
            bat_cap,
            batch_size=32,
            learning_rate=0.001,
            criterion=torch.nn.MSELoss(),
            min_beta=0,
            max_beta=1
            ):
        """
        The training class for the pytorch model
        :param model: The model that we train
        :param X_train: the tensor with training values for X
        :param y_train: the tensor with training values for y
        :param X_test: the tensor with test values for X
        :param y_test: the tensor with test values for y
        :param epochs: the number of epochs that we wish to train for
        :param batch_size: the batch size before going through backpropagation
        :param learning_rate: the learning rate
        :param criterion: the criterion by which to evaluate the performance (i.e. the loss function)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.min_beta = min_beta
        self.max_beta = max_beta

        train_data = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        test_data = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        self.train_loader = DataLoader(train_data, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        self.model = model
        self.cvx = cvx_layer
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.T = T
        self.bat_cap = bat_cap

        days = y_train.shape[0] + y_test.shape[0]
        self.months = round(days / 30.5)


    def fit(self, verbose=0, timer=False):
        """
        The training loop itself with error handling for cvx layer issues.
        :return: state_dict_list: the state dictionary for each of the epochs, argmin_test: the best epoch
        """
        avg_train_error = []
        avg_train_mse = []
        avg_train_regret = []
        avg_test_error = []
        avg_test_mse = []
        avg_test_regret = []
        state_dict_list = []
        beta = self.min_beta

        try:
            for epoch in range(self.epochs):
                num_train_batches = 0
                num_test_batches = 0
                total_loss = 0
                total_mse = 0
                total_regret = 0
                total_loss_test = 0
                total_mse_test = 0
                total_regret_test = 0

                # Initialize timers
                total_model_time = 0
                total_cvx_time = 0
                total_backprop_time = 0

                self.model.train()
                batches = iter(self.train_loader)

                for input, output in batches:
                    model_start = time.perf_counter()

                    pv_train, y_train = self.model(input[:, :, 0:-4],
                                                   input[:, -self.T:, -4],
                                                   input[:, -self.T:, -3],
                                                   input[:, -self.T:, -2],
                                                   input[:, -1, -1])

                    total_model_time += time.perf_counter() - model_start

                    cvx_start = time.perf_counter()
                    y_train = self.cvx(output[:, :, 0],
                                       input[:, -self.T:, -4],
                                       input[:, -self.T:, -3],
                                       input[:, -self.T:, -2],
                                       input[:, -1, -1],
                                       y_train[-2],
                                       y_train[-1])
                    total_cvx_time += time.perf_counter() - cvx_start

                    '''
                    prediction = (torch.bmm(y_train[0].unsqueeze(1), input[:, -self.T:, -3].unsqueeze(-1)) -
                                  torch.bmm(y_train[1].unsqueeze(1), input[:, -self.T:, -2].unsqueeze(-1)))
                    prediction = prediction.squeeze([1, 2])
                    '''

                    prediction = torch.sub(torch.mul(y_train[0], input[:, -self.T:, -3]),
                                           torch.mul(y_train[1], input[:, -self.T:, -2]))

                    mse_loss = self.criterion(pv_train, output[:, :, 0])
                    regret = self.criterion(prediction, output[:, :, 1])

                    loss = (beta * regret + (1 - beta) * mse_loss)

                    total_loss += float(loss)
                    total_mse += float(mse_loss)
                    total_regret += float(regret)

                    num_train_batches += 1

                    backprop_start = time.perf_counter()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_backprop_time += time.perf_counter() - backprop_start

                self.model.eval()

                with torch.inference_mode():
                    test_batches = iter(self.test_loader)

                    for input, output in test_batches:
                        pv_test, y_test = self.model(input[:, :, 0:-4],
                                                     input[:, -self.T:, -4],
                                                     input[:, -self.T:, -3],
                                                     input[:, -self.T:, -2],
                                                     input[:, -1, -1])

                        y_test = self.cvx(output[:, :, 0],
                                          input[:, -self.T:, -4],
                                          input[:, -self.T:, -3],
                                          input[:, -self.T:, -2],
                                          input[:, -1, -1],
                                          y_test[-2],
                                          y_test[-1])

                        '''
                        prediction = (torch.bmm(y_test[0].unsqueeze(1), input[:, -self.T:, -3].unsqueeze(-1)) -
                                      torch.bmm(y_test[1].unsqueeze(1), input[:, -self.T:, -2].unsqueeze(-1)))
                        prediction = prediction.squeeze([1, 2])
                        '''

                        prediction = torch.sub(torch.mul(y_test[0], input[:, -self.T:, -3]),
                                               torch.mul(y_test[1], input[:, -self.T:, -2]))

                        mse_loss = self.criterion(pv_test, output[:, :, 0])
                        regret = self.criterion(prediction, output[:, :, 1])

                        test_loss = (beta * regret + (1 - beta) * mse_loss)

                        total_loss_test += float(test_loss)
                        total_mse_test += float(mse_loss)
                        total_regret_test += float(regret)
                        num_test_batches += 1

                avg_train_error.append(total_loss / num_train_batches)
                avg_train_mse.append(total_mse / num_train_batches)
                avg_train_regret.append(total_regret / num_train_batches)
                avg_test_error.append(total_loss_test / num_test_batches)
                avg_test_mse.append(total_mse_test / num_test_batches)
                avg_test_regret.append(total_regret_test / num_test_batches)

                state_dict_list.append(self.model.state_dict())

                if timer is True:
                    print(f"Epoch model time: {total_model_time:.2f} seconds")
                    print(f"Epoch cvx time: {total_cvx_time:.2f} seconds")
                    print(f"Epoch backprop time: {total_backprop_time:.2f} seconds")

                if epoch % 5 == 0 and verbose >= 1:
                    print('Step {}:\n'
                          'Average train loss: {:.4f} | Average test loss: {:.4f}\n'
                          '   Regret: {:.4f} | {:.4f}\n'
                          '   MSE: {:.4f} | {:.4f}\n'
                          .format(epoch,avg_train_error[epoch],avg_test_error[epoch],avg_train_error[epoch],avg_test_regret[epoch],avg_train_mse[epoch],avg_test_mse[epoch]))

                if epoch % (self.epochs / 10) == 0 and verbose >= 1:
                    beta = self.min_beta + (self.max_beta - self.min_beta) * (epoch / (self.epochs - self.epochs / 10))
                    print(f'MSE: {1 - beta:.2f} | Regret: {beta:.2f}')

        except Exception as e:
            print(f"Training interrupted due to error: {e}")
            print("Returning progress up to this point...")

        argmin_test = avg_test_error.index(min(avg_test_error)) if avg_test_error else -1

        print('Best Epoch: ' + str(argmin_test))

        if verbose == 2:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

            axes[0].plot(avg_train_error, label='train error')
            axes[0].plot(avg_test_error, label='test error')
            axes[0].set_title('Error')
            axes[0].legend()

            axes[1].plot(avg_train_mse, label='train mse')
            axes[1].plot(avg_test_mse, label='test mse')
            axes[1].set_title('MSE')
            axes[1].legend()

            axes[2].plot(avg_train_regret, label='train regret')
            axes[2].plot(avg_test_regret, label='test regret')
            axes[2].set_title('Regret')
            axes[2].legend()

            plt.tight_layout()
            plt.show()

        return state_dict_list, argmin_test

