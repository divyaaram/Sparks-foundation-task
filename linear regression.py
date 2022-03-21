{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "dd70e2b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#PREDICTION USING SUPERVISED ML\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "219ac96f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np  \n",
    "import matplotlib.pyplot as plt  \n",
    "%matplotlib inline\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a6592639",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data loaded successfully\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.5</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>9.2</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5.5</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.3</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2.7</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30\n",
       "5    1.5      20\n",
       "6    9.2      88\n",
       "7    5.5      60\n",
       "8    8.3      81\n",
       "9    2.7      25"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "url = \"http://bit.ly/w-data\"\n",
    "s_data = pd.read_csv(url)\n",
    "print(\"Data loaded successfully\")\n",
    "\n",
    "s_data.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3bc63365",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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WTtmzlvMD259mhdWrGarAU2cNmEkE8d0ZiiK7lOziR9w0i/iz8Kses2ct5hv37CA1W9lFb8Xr1jNt29YAFCR5O+Tu2ZmZfaDW59+N+kXrH7rbX5w69MViceJfz2dd9557LLLLuy2226MHj2ahx56qNIhmVmVeWHF6k4tL7ea7urpjHL0rz3wwAPcdNNNzJ07l4022oiXXnqJN998s8v7W7t2Lb17N8w/iVnD2GpAE4vbSPJbDWiqQDQN0uIv9K8tXrGa4L3+tZnzFne4bSlLlixhiy22YKONssGRtthiC7baaisefvhhPvrRj7L77rszduxYVq5cyZo1azjuuOMYNWoUY8aM4a677gLgsssu46ijjuKwww7j4IMP5vXXX+dLX/oSe+65J2PGjOHGG28E4PHHH2fs2LGMHj2a3XbbjWeffXa9YjeznnPahJE09em1zrKmPr04bUKpQdrKpyGal6X619an1X/wwQdz7rnn8qEPfYgDDzyQyZMnM27cOCZPnszVV1/NnnvuyWuvvUZTUxM/+clPAFiwYAFPPfUUBx98MM888wyQ/XJ49NFHGThwIGeeeSb7778/l1xyCStWrGDs2LEceOCBXHjhhZxyyikcc8wxvPnmm7z9dqeGhTWzCirkGV/V04PK1b+2ySabMGfOHO677z7uuusuJk+ezHe+8x2GDBnCnnvuCUD//v0BuP/++5k6dSoAO+20E9tuu+27if+ggw5i4MCBANx2223MmjWLH/4wGzp2zZo1LFy4kHHjxnHeeeexaNEiJk2axIgRI9YrdjPrWRPHDK1Yom+tIRJ/OfvXevXqxfjx4xk/fjyjRo3iggsuaPPSylID3my88cbrrHf99dczcuS6PwF33nln9tprL26++WYmTJjARRddxP7777/e8ZtZ42mIPv5y9a89/fTT6/S1z58/n5133pkXXniBhx9+GICVK1eydu1a9t13X6644goAnnnmGRYuXPi+5A4wYcIEfvazn737RTFvXjYK3l/+8he23357Tj75ZA4//HAeffTR9YrdzBpXQ7T4y9W/tmrVKqZOncqKFSvo3bs3O+64I9OnT+e4445j6tSprF69mqamJm6//XZOPPFETjjhBEaNGkXv3r257LLL3j0pXOzf/u3fOPXUU9ltt92ICIYPH85NN93E1VdfzeWXX06fPn344Ac/yFlnnbVe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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "s_data.plot(x='Hours', y='Scores', style='o')  \n",
    "plt.title('Hours vs Percentage')  \n",
    "plt.xlabel('Hours Studied by the student')  \n",
    "plt.ylabel('Percentage Score obtained')  \n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6f4f1860",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = s_data.iloc[:, :-1].values  \n",
    "y = s_data.iloc[:, 1].values  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8c0968d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split  \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, \n",
    "                            test_size=0.2, random_state=0) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f6326a44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training complete.\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression  \n",
    "regressor = LinearRegression()  \n",
    "regressor.fit(X_train, y_train) \n",
    "\n",
    "print(\"Training complete.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "fdd4608a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "line = regressor.coef_*X+regressor.intercept_\n",
    "plt.scatter(X, y)\n",
    "plt.plot(X, line);\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "4a876160",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.5]\n",
      " [3.2]\n",
      " [7.4]\n",
      " [2.5]\n",
      " [5.9]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)\n",
    "y_pred = regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "87d8722f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>16.884145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>27</td>\n",
       "      <td>33.732261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69</td>\n",
       "      <td>75.357018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>26.794801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60.491033</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Actual  Predicted\n",
       "0      20  16.884145\n",
       "1      27  33.732261\n",
       "2      69  75.357018\n",
       "3      30  26.794801\n",
       "4      62  60.491033"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  \n",
    "df \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "61bf42b1",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<function matplotlib.pyplot.show(close=None, block=None)>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.plot(kind='bar')\n",
    "plt.title(\"Actual Vs Predicted\")\n",
    "plt.show\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5470aefa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No of Hours = 9.25\n",
      "Predicted Score = 93.69173248737539\n"
     ]
    }
   ],
   "source": [
    "hours = 9.25\n",
    "test = np.array([hours])\n",
    "test = test.reshape(-1,1)\n",
    "pred = regressor.predict([[hours]])\n",
    "print(\"No of Hours = {}\".format(hours))\n",
    "print(\"Predicted Score = {}\".format(pred[0]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "5078e6a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 4.183859899002982\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics  \n",
    "print('Mean Absolute Error:', \n",
    "      metrics.mean_absolute_error(y_test, y_pred)) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d734cb3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
