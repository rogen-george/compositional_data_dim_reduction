from load_data_news_groups import get_data_news_groups
from load_data_diet_swap import get_data_diet_swap
from load_data_atlas import get_data_atlas

experiments = ["news_groups", "diet_swap", "atlas"]
experiment = "news_groups"

if experiment == "diet_swap":
    positive_train, negative_train, validation_data, validation_label, test_data, test_label = get_data_diet_swap()
    sample_range = range( 20, 80, 10 )
    k_values = [5, 10, 20]
    save_folder = "diet_swap"


elif experiment == "news_groups":
    save_folder = "accuracy_roc"
    sample_range = range(10, 1000, 100)
    k_values = [50]
    positive_train, negative_train, validation_data, validation_label, test_data, test_label =  get_data_news_groups()

elif experiment == "atlas":
    save_folder = "atlas"
    sample_range = range( 20, 80, 10 )
    k_values = [5, 10, 20]
    positive_train, negative_train, validation_data, validation_label, test_data, test_label = get_data_atlas()