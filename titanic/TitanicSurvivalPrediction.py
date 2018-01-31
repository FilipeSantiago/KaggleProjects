import numpy as np
import pandas
from Validation import NFoldCrossValidation as NFold
from DataManipulation import DataManipulation as Dm
from MachineLearning.Classifiers import ClassifierFactory
from MachineLearning.Classifiers import ClassifierTypes
from Graphics import BarChart
import matplotlib.pyplot as plt
import seaborn as sns


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

    def plot_relation_attribute_x_class(self, attribute, data_frame=None, labels=None):

        if data_frame is None:
            data_frame = self.df

        bar_chart_plotter = BarChart.BarChart()
        bar_chart_plotter.plot_chart(data_frame[attribute], self.get_classes(), labels)

    def calc_accuracy(self):
        self.df = self.data_manipulator(self.df)

        factory = ClassifierFactory.ClassifierFactory()

        learning_algorithm = factory.choose_classifier(self.classifier_types.Tree)

        learning_algorithm.fit(self.get_attributes(), self.get_classes())

        n_fold = NFold.NFoldCrossValidation(learning_algorithm, self.get_attributes(), self.get_classes())
        accuracy = n_fold.ten_fold_cross_validation()

        self.prediction_csv(learning_algorithm, self.data_manipulator, 'test', 'PassengerId', 'kaggle003')
        print(accuracy)

    def prediction_csv(self, learning_algorithm, data_manipulator, test_csv, columns_in_output, output_name):

        test_df = pandas.read_csv('input_data/' + test_csv + '.csv', sep=',')

        ids = test_df[columns_in_output]
        test_df = data_manipulator(test_df)

        classes = learning_algorithm.predict(test_df)

        response_data = {'PassengerId': ids, 'Survived': classes}
        response_df = pandas.DataFrame(data=response_data)
        response_df.to_csv('output_data/' + output_name + '.csv', sep=',')

    def data_manipulator(self, data_frame):
        data_manipulation = Dm.DataManipulation(data_frame)

        self.build_new_attribute_based_on_ticket_value(data_frame)
        # self.build_new_attribute_based_on_same_ticket(data_frame)
        self.build_new_attribute_based_on_title(data_frame)
        self.build_new_attribute_based_on_have_a_cabin(data_frame)
        # self.build_new_attribute_based_on_family_name(data_frame)
        self.build_new_attribute_based_on_family_and_friends(data_frame)

        data_range = [-np.inf, 10.5, 18.5, 35.5, 55.5, 65.5, 80.5, np.inf]
        data_range1 = [-np.inf, 0.5, 2.5, np.inf]
        data_range2 = [-np.inf, 1.5, 4.5, np.inf]
        data_range3 = [-np.inf, 0.5, np.inf]

        data_manipulation.drop_columns(['Name', 'Ticket', 'Embarked', 'Cabin', 'PassengerId', 'Fare', 'SibSp', 'Parch'])\
                         .set_categorical_columns(['Sex'])\
                         .discretize_column('Age', data_range) \
                         .discretize_column('FriendsNumber', data_range3) \
                         .discretize_column('FamilyNumber', data_range2) \
        #                .discretize_column('SibSp', data_range1) \

        data_frame = data_manipulation.get_data_frame()
        self.plot_relation_attribute_x_class('FamilyNumber', data_frame)

        print(data_frame.keys())
       # print(data_frame['SameSurname'].corr(data_frame['SameTicket']))

       # correlation_matrix = data_frame.corr()
       # plt.figure(figsize=(10, 8))
       # ax = sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='RdYlGn')
       # plt.title('Correlation matrix between the  features')
       # plt.show()

        data_frame['Age'].fillna(len(data_range) - 2, inplace=True)

        return data_frame

    @staticmethod
    def build_new_attribute_based_on_title(data_frame):

        titles = ['Miss', 'Mme', 'Mlle', 'Master', 'Ms', 'Mr', 'Mrs', 'Rev', 'Major', 'Lady', 'Sir', 'Dona',
                   'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dr', 'Don']

        miss = ['Miss', 'Mlle']
        mrs = ['Mrs', 'Ms', 'Mme']
        mr = ['Mr', 'Master']
        rare_titles = ['Rev', 'Major', 'Col', 'Capt', 'Dr', 'Lady', 'Sir', 'the Countess', 'Jonkheer', 'Don', 'Dona']

        labels = ['miss', 'mrs', 'mr', 'rare_titles']
        title_categories = [miss, mrs, mr, rare_titles]

        names = data_frame.Name
        categories = []

        for name in names:
            category = ''

            for idx in range(len(titles)):
                if (titles[idx] + '.') in name:
                    category = titles[idx]
                    #categories.append(idx)
                    break

            if category == '':
                break
            else:
                for idx in range(len(title_categories)):
                    title_category = title_categories[idx]
                    if category in title_category:
                        categories.append(idx)
                        break

        # print(len(categories))
        # print(len(data_frame['Survived']))
        data_frame['Titles'] = categories

        #test.plot_relation_attribute_x_class('Titles', labels)

    @staticmethod
    def build_new_attribute_based_on_ticket_value(data_frame):

        mask_class = (data_frame.Pclass == 1) & (data_frame.Fare > 300) & (data_frame.Fare < 5000)
        data_frame.ix[mask_class, 'Pclass'] = 0

    @staticmethod
    def build_new_attribute_based_on_have_a_cabin(data_frame):
        cabins = data_frame.Cabin
        cabin_class = data_frame.Pclass

        has_cabin = []
        for idx in range(len(cabins)):
            if not isinstance(cabins[idx], float) and cabin_class[idx] <= 0:
                has_cabin.append(1)
            else:
                has_cabin.append(0)

        data_frame['HasCabin'] = has_cabin

    @staticmethod
    def build_new_attribute_based_on_same_ticket(data_frame):

        tickets = list(data_frame.Ticket)
        number_of_tickets = []
        ticket_count = {}

        for ticket in tickets:
            if ticket not in ticket_count:
                ticket_count[ticket] = tickets.count(ticket)
            number_of_tickets.append(ticket_count[ticket])

        data_frame['SameTicket'] = number_of_tickets

    def build_new_attribute_based_on_family_and_friends(self, data_frame):

        relations = data_frame[['Name', 'SibSp', 'Parch', 'Ticket']]

        surnames = list(map(self.take_family_name, relations.Name))
        relations['Surname'] = surnames

        # ('SibSp' + 'Parch' + 1)
        suposed_family_number = []
        family_with_same_ticket = []
        friends_with_same_ticket = []

        for relation in relations.iterrows():

            relation_suposed_family_number = relation[1]['SibSp'] + relation[1]['Parch'] + 1

            suposed_family_number.append(relation_suposed_family_number)

            ticket_mask_class = (relations.Ticket == relation[1].Ticket)
            ticket_data_frame = relations.loc[ticket_mask_class]

            surname_mask_class = (ticket_data_frame.Surname == relation[1].Surname)
            surname_and_ticket_data_frame = ticket_data_frame.loc[surname_mask_class]

            family_same_ticket = len(surname_and_ticket_data_frame['Surname'])
            different_surnames = set(ticket_data_frame['Surname'])
            same_surname = len(different_surnames)

            family_with_same_ticket.append(family_same_ticket)
            friends_with_same_ticket.append(same_surname - 1)

        data_frame['FamilyNumber'] = suposed_family_number
        data_frame['FriendsNumber'] = friends_with_same_ticket

    def build_new_attribute_based_on_family_name(self, data_frame):

        names = list(data_frame.Name)
        surnames = list(map(self.take_family_name, names))
        number_of_surnames = []
        surnames_count = {}

        for surname in surnames:
            if surname not in surnames_count:
                surnames_count[surname] = surnames.count(surname)
            number_of_surnames.append(surnames_count[surname])

        data_frame['SameSurname'] = number_of_surnames

    @staticmethod
    def take_family_name(name):
        return name.partition(",")[0]



test = TitanicSurvivalPrediction('input_data/train.csv', 'Survived')
test.calc_accuracy()
