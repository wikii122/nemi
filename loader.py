"""
Set of functions used to load data.
"""
import csv


def from_csv(path, ignore_units=True):
    """
    Load given CSV file and return dict containing its contents under names
    found in first row.

    Notes:
        - file correctness is not checked
        - delimiter: ','
        - quote: '"'
        - file must have table names in first row
        - is ignore_units is true, second row is ignored.
    """
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        data = {}
        headers = next(reader)
        if ignore_units:
            units = next(reader)
        for row in reader:
            for item, name in zip(row, headers):
                if not name in data:
                    data[name] = list()
                data[name].append(item)

    return data
