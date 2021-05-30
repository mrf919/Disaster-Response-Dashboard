import sys
# import libraries
import pandas as pd

# the function for loading data from defined path (data extraction)
def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on=('id'))
    return df

# function for data cleaning
def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';',expand=True )
    categories = categories.rename(columns=categories.iloc[0])
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1:])
        categories[column] = categories[column].astype('int64')
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df=df.drop_duplicates()
    return df
# fuction for saving the data in sqlite data base on the given path
def save_data(df, database_filepath):
    #Save the clean dataset into an sqlite database
    from sqlalchemy import create_engine
    eng = 'sqlite:///' + database_filepath
    engine = create_engine(eng)
    df.to_sql('message', engine, index=False)  

# the main function including the ETL Pipeline, beginning with the file pathes
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
