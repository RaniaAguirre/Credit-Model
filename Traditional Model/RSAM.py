import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Cambia el backend a TkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


class RSAMScoring:
    def __init__(self, df):
        self.df = df
        self.poor_std_ranges = {
            'Credit_History_Age_Months': [
                (-np.inf, 148, 102),
                (148, 185, 102),
                (185, 232, 99),
                (232, 251, 98),
                (251, np.inf, 96),
                ('<missing>', None)
            ],
            'Num_Bank_Accounts': [
                (-np.inf, 6, 99),
                (6, 7, 100),
                (7, 9, 100),
                (9, np.inf, 101),
                ('<missing>', None)
            ],
            'Num_of_Loan': [
                (-np.inf, 2, 98),
                (2, 5, 99),
                (5, 7, 101),
                (7, np.inf, 101),
                ('<missing>', None)
            ],
            'Outstanding_Debt': [
                (-np.inf, 520.99, 127),
                (520.99, 788.4, 127),
                (788.4, 1058.68, 126),
                (1058.68, 1290.83, 120),
                (1290.83, 1468.67, 101),
                (1468.67, np.inf, 75),
                ('<missing>', None)
            ],
            'Delay_from_due_date': [
                (-np.inf, 7, 107),
                (7, 14, 106),
                (14, 18, 102),
                (18, 21, 100),
                (21, 28, 100),
                (28, 34, 99),
                (34, 49, 90),
                (49, np.inf, 89),
                ('<missing>', None)
            ],
            'Num_Credit_Inquiries': [
                (-np.inf, 2, 109),
                (2, 6, 109),
                (6, 7, 99),
                (7, 8, 99),
                (8, 9, 98),
                (9, 11, 91),
                (11, np.inf, 91),
                ('<missing>', None)
            ]
        }
        self.std_good_ranges = {
            'Credit_History_Age_Months': [
                (-np.inf, 160, 93),
                (160, 195, 111),
                (195, 237, 114),
                (237, 269, 120),
                (269, 341, 122),
                (341, np.inf, 124),
                ('<missing>', None)
            ],
            'Num_Bank_Accounts': [
                (-np.inf, 3, 135),
                (3, 4, 119),
                (4, 5, 119),
                (5, 6, 118),
                (6, 7, 103),
                (7, 9, 101),
                (9, np.inf, 96),
                ('<missing>', None)
            ],
            'Num_Credit_Card': [
                (-np.inf, 2, 158),
                (2, 3, 157),
                (3, 5, 119),
                (5, 6, 117),
                (6, 8, 107),
                (8, np.inf, 94),
                ('<missing>', None)
            ],
            'Num_of_Loan': [
                (-np.inf, 1, 117),
                (1, 2, 117),
                (2, 4, 117),
                (4, 5, 117),
                (5, np.inf, 111),
                ('<missing>', None)
            ],
            'Delay_from_due_date': [
                (-np.inf, 5, 133),
                (5, 8, 123),
                (8, 10, 122),
                (10, 15, 122),
                (15, 19, 113),
                (19, 23, 105),
                (23, 27, 104),
                (27, 32, 101),
                (32, np.inf, 89),
                ('<missing>', None)
            ],
            'Num_Credit_Inquiries': [
                (-np.inf, 2, 120),
                (2, 4, 119),
                (4, 5, 118),
                (5, 6, 115),
                (6, 7, 113),
                (7, 9, 111),
                (9, np.inf, 109),
                ('<missing>', None)
            ]
        }

    def get_score(self, value, ranges):
        """Obtain score for a given value based on specified ranges."""
        for r in ranges:
            if r[0] == '<missing>' and pd.isna(value):
                return r[2]
            if isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)):
                if r[0] <= value < r[1]:
                    return r[2]
        return None

    def calculate_poor_std_points(self):
        """Calculate the poor_std_points based on ranges."""
        # Inicializamos una columna para almacenar los puntos totales por cliente
        self.df['poor_std_points_total'] = 0

        # Iteramos sobre las columnas y rangos de `poor_std_ranges`
        for column, ranges in self.poor_std_ranges.items():
            self.df[f'{column}_poor_std_points'] = self.df[column].apply(lambda x: self.get_score(x, ranges))

            # Sumamos los puntos de la columna correspondiente para cada cliente
            self.df['poor_std_points_total'] += self.df[f'{column}_poor_std_points']

    def calculate_std_good_points(self):
        """Calculate the std_good_points based on ranges."""
        # Inicializamos una columna para almacenar los puntos totales por cliente
        self.df['std_good_points_total'] = 0

        # Iteramos sobre las columnas y rangos de `std_good_ranges`
        for column, ranges in self.std_good_ranges.items():
            self.df[f'{column}_std_good_points'] = self.df[column].apply(lambda x: self.get_score(x, ranges))

            # Sumamos los puntos de la columna correspondiente para cada cliente
            self.df['std_good_points_total'] += self.df[f'{column}_std_good_points']

    def calculate_final_points(self):
        """Calculate final points based on the poor_std_points and std_good_points."""
        # Aquí elegimos el puntaje total correspondiente
        self.df['final_points'] = np.where(self.df['poor_std_points_total'] < 600,
                                           self.df['poor_std_points_total'],
                                           self.df['std_good_points_total'])

        predicted_category = ['Poor' if points < 600 else 'Standard' if points <= 725 else 'Good'
                              for points in self.df['final_points']]

        self.df['predicted_category'] = predicted_category

    def plot_histograms(self, variable):
        """Generate a histogram for the specified numerical feature."""
        target = 'Credit_Score'

        if variable not in self.df.columns:
            print(f"Error: '{variable}' is not a valid column.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[self.df[target] == 0][variable], color='red', label='Credit_Score = 0', kde=True, bins=30,
                     stat='density')
        sns.histplot(self.df[self.df[target] == 1][variable], color='blue', label='Credit_Score = 1', kde=True, bins=30,
                     stat='density')
        sns.histplot(self.df[self.df[target] == 2][variable], color='green', label='Credit_Score = 2', kde=True,
                     bins=30,
                     stat='density')
        plt.title(f'Distribution of {variable} by Credit_Score')
        plt.xlabel(variable)
        plt.ylabel('Density')
        plt.legend(title='Credit_Score')
        plt.grid(True)
        plt.show()

    def confusion_heatmap(self):
        """Classify final points into categories."""

        true_labels = self.df['Score_Category'].astype('category')
        pred_labels = pd.Categorical(self.df['predicted_category'])

        accuracy = accuracy_score(true_labels, pred_labels)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Poor', 'Standard', 'Good'],
                    yticklabels=['Poor', 'Standard', 'Good'])
        plt.title('Confusion Matrix: True vs Predicted Categories')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


# Example usage:
df = pd.read_csv('train_cleaned.csv') # Change for your data set
scoring = RSAMScoring(df)
scoring.calculate_poor_std_points()
scoring.calculate_std_good_points()
scoring.calculate_final_points()
#scoring.plot_histograms('final_points')  # Indicate the variable you want to analyze
#scoring.confusion_heatmap()
print(df)


