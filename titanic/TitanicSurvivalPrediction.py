import numpy as np
import pandas
from Validation import NFoldCrossValidation as NFold
from DataManipulation import DataManipulation as Dm
from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes


class TitanicSurvivalPrediction:

    def __init__(self, csv_name, class_name):
        self.df = pandas.read_csv(csv_name, sep=',')
        self.class_name = class_name
        self.data_manipulation = Dm.DataManipulation(self.df)
        self.classifier_types = ClassifierTypes.ClassifierTypes
        self.learning_algorithm = None

    def get_attributes(self):
        return self.df.drop(self.class_name, 1)

    def get_classes(self):
        return self.df[self.class_name]

    def data_manipulator(self):

        self.build_new_class_based_on_ticket_value()

        self.data_manipulation.drop_columns(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'])\
                              .set_categorical_columns(['Sex'])\
                              .discretize_column('Age', [-np.inf, 0, 10, 18, 35, 55, 65, 80, np.inf])

        self.df = self.data_manipulation.get_data_frame()
        self.df['Age'].fillna(7, inplace=True)

    def build_new_class_based_on_ticket_value(self):

        mask_class = (self.df.Pclass == 1) & (self.df.Fare > 300)

        self.df.ix[mask_class, 'Pclass'] = 0

        self.data_manipulation.drop_columns(['Fare'])

    def make_prediction(self):

        self.data_manipulator()

        factory = ClassifierFactory.ClassifierFactory()

        learning_algorithm = factory.choose_classifier(self.classifier_types.SVM)

        n_fold = NFold.NFoldCrossValidation(learning_algorithm, self.get_attributes(), self.get_classes())
        accuracy = n_fold.ten_fold_cross_validation()

        print(accuracy)


test = TitanicSurvivalPrediction('input_data/train.csv', 'Survived')
test.make_prediction()
