import pandas

# read dataset/dataset_filtrado.csv
df = pandas.read_csv("./dataset/dataset_filtrado.csv", sep=";")  # Example for semicolon

print(df["URL"])  # Print the column "url"
