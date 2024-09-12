def histogram(data, n, b, h):
    # data is a list
    # n is an integer
    # b and h are floats
    # Write your code here
    if b == h:
        print('b and h are the same value')
        return []
    elif b > h:
        temp = b
        b = h
        h = temp
    if n == 0:
        return []
    
    hist = n * [0]
    w = (h - b) / n
    # go through each element and each bin
    for c in range(len(hist)):
        for i in range(len(data)):
            # check which bound it is in
            # check if it is in first bin
            if data[i] > b and data[i] < b + w:
                hist[c] += 1
            else:   
                # for second to last bin, check which bin the value is in
                if data[i] >= b + c * w and data[i] < b + (c + 1) * w:
                    hist[c] += 1
    return hist
            
    # return the variable storing the histogram
    # Output should be a list



def happybirthday(name_to_day, name_to_month, name_to_year):
    #name_to_day, name_to_month and name_to_year are dictionaries
    # Create a dictonary of months that give name, (day, year, age) <- tuple
    # Write your code here
    month_to_all = {}

    # for each name
    for name in name_to_day:
    # get day, month, year
        day = name_to_day[name]
        month = name_to_month[name]
        year = name_to_year[name]
    # caculate age
        age = 2022 - year
    # put into month_to_all
        month_to_all[str(month)] = (name, (day, year, age))

    return(month_to_all)
    
    # return the variable storing month_to_all
    # Output should be a dictionary