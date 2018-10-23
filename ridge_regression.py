import numpy as np
import scipy.io as sio

data_Mat = sio.loadmat('dataset2.mat')


def phi(x, n):
    return np.array([x ** i for i in range(0, n + 1)])


def split_dataset(x, y, split):
    split_x = np.vsplit(x, split)
    split_y = np.vsplit(y, split)
    return split_x, split_y


def gradient_descent(x, y, learning_rate, lam, iters, epsilon):
    theta = np.zeros((x.shape[1], 1))

    x_trans = np.transpose(x)
    for i in range(0, iters):
        x_theta = np.matmul(x, theta)
        xTminusy = np.subtract(x_theta, y)
        partial_deri = np.matmul(x_trans, xTminusy) * 2
        lamb_theta = lam * theta * 2
        partial_deri2 = np.add(partial_deri, lamb_theta)
        partial_deri2 = partial_deri2 * learning_rate
        new_theta = theta - partial_deri2
        diff = new_theta - theta
        if np.linalg.norm(diff, 2) <= epsilon:
            theta = new_theta
            break
        theta = new_theta
    return theta


def calculate_loss(theta, lam, x_holdout, y_holdout):
    x_theta = np.matmul(x_holdout, theta)
    term1 = y_holdout - x_theta
    theta_norm = np.linalg.norm(theta, 2) ** 2
    lam_theta = lam * theta_norm
    temp = np.linalg.norm(term1, 2) ** 2
    return lam_theta + temp


lambda_exp = [0.1, 0.001, 0.0001, 1, 2, 0.00001, 0.000001, 3, 0.01, 0.00000000000001]


def ridge_regress(x_t, y_t):
    master_holdout_error = []
    for j in lambda_exp:
        master_loss = []
        for i in range(len(x_t)):
            x_train_copy = x_t.copy()
            del x_train_copy[i]
            y_copy = y_t.copy()
            del y_copy[i]
            x_train_copy = np.concatenate(x_train_copy)
            y_copy = np.concatenate(y_copy)
            master_loss.append(
                calculate_loss(gradient_descent(x_train_copy, y_copy, 0.00000001, j, 60, 0.001), j, x_t[i], y_t[i]))
        avg = np.average(np.array(master_loss))
        master_holdout_error.append(avg)
    final_lambda = np.argmin(np.array(master_holdout_error))
    return final_lambda


x_train, x_test, y_train, y_test = data_Mat['X_trn'], data_Mat['X_tst'], data_Mat['Y_trn'], data_Mat['Y_tst']
# print(data_Mat)
x_train2 = np.array([phi(x_train[i][0], 2) for i in range(0, len(x_train))])
# print(x_test.shape)
# exit(0)
x_test2 = np.array([phi(x_test[i][0], 2) for i in range(0, len(x_test))])
x_train5 = np.array([phi(x_train[i][0], 5) for i in range(0, len(x_train))])
x_test5 = np.array([phi(x_test[i][0], 5) for i in range(0, len(x_test))])

splitted2_2 = split_dataset(x_train2, y_train, 2)
x_train2_2 = splitted2_2[0]
y_train_2 = splitted2_2[1]

splitted2_N = split_dataset(x_train2, y_train, len(x_train2))
x_train2_N = splitted2_N[0]
y_train_N = splitted2_N[1]

splitted5_2 = split_dataset(x_train5, y_train, 2)
x_train5_2 = splitted5_2[0]
y_train_2 = splitted5_2[1]

splitted5_N = split_dataset(x_train5, y_train, len(x_train5))
x_train5_N = splitted5_N[0]
y_train_N = splitted5_N[1]

lam2_2 = ridge_regress(x_train2_2, y_train_2)

lam2_N = ridge_regress(x_train2_N, y_train_N)

lam5_2 = ridge_regress(x_train5_2, y_train_2)

lam5_N = ridge_regress(x_train5_N, y_train_N)

theta2_2 = gradient_descent(np.concatenate(x_train2_2), np.concatenate(y_train_2), 0.00000001, lambda_exp[lam2_2], 60,
                            0.0010)

theta2_N = gradient_descent(np.concatenate(x_train2_N), np.concatenate(y_train_N), 0.00000001, lambda_exp[lam2_N], 60,
                            0.0010)

theta5_2 = gradient_descent(np.concatenate(x_train5_2), np.concatenate(y_train_2), 0.00000000001, lambda_exp[lam5_2], 70,
                            0.0001)

theta5_N = gradient_descent(np.concatenate(x_train5_N), np.concatenate(y_train_N), 0.00000000001, lambda_exp[lam5_N], 70,
                            0.0001)

print(calculate_loss(theta2_2, lambda_exp[lam2_2], x_test2, y_test))
print(calculate_loss(theta2_2, lambda_exp[lam2_2], np.concatenate(x_train2_2), np.concatenate(y_train_2)))
print(lambda_exp[lam2_2])
print(theta2_2)

print("-------------------------------------")

print(calculate_loss(theta2_N, lambda_exp[lam2_N], x_test2, y_test))
print(calculate_loss(theta2_N, lambda_exp[lam2_N], np.concatenate(x_train2_N), np.concatenate(y_train_N)))
print(lambda_exp[lam2_N])
print(theta2_N)

print("-------------------------------------")

print(calculate_loss(theta5_2, lambda_exp[lam5_2], x_test5, y_test))
print(calculate_loss(theta5_2, lambda_exp[lam5_2], np.concatenate(x_train5_2), np.concatenate(y_train_2)))
print(lambda_exp[lam5_2])
print(theta5_2)

print("-------------------------------------")

print(calculate_loss(theta5_N, lambda_exp[lam5_N], x_test5, y_test))
print(calculate_loss(theta5_N, lambda_exp[lam5_N], np.concatenate(x_train5_N), np.concatenate(y_train_N)))
print(lambda_exp[lam5_N])
print(theta5_N)
