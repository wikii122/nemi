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

def normalize(dict):
	"""
	Normalizes data in given dict to be centered around 0 and have a std deviation of 1.

	Please note data after analysis should be denormalized.
	"""

	ret = {}

	average = get_average(dict)
	variance = get_variance(dict, average)

	for key, list in dict:
		ret[key] = []
		for element in list:
			val = element - average[key]
			val /= variance[key]**0.5
			ret[key].append(val)

	return ret

def get_average(dict):
	"""
	Return the average value for every entry in the dict
	"""
	
	ret = {}

	for key, list in dict:
		ret[key] = 0
		num = 0

		for element in list:
			ret[key] += element
			num += 1

		if num > 0:
			ret[key] /= num

	return ret

def get_variance(dict, average = None): 
	"""
	Returns the variance of data in the dict
	"""
	if average is None:
		average = get_average(dict)

	ret = {}

	for key, list in dict:
		ret[key] = 0
		num = 0

		for element in list:
			ret[key] += (element - average[key])**2
			num += num

		if num > 0:
			ret[key] /= num

	return ret

