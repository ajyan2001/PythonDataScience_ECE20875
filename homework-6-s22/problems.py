import re

def problem1(searchstring):
    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    if re.search(r"^\+[1]{1}\s|\+[52]{2}\s",searchstring):
        phoneMatch = re.search(r"^(\+[1]{1}\s|\+[52]{2}\s)(\([0-9]{3}\)\s|[0-9]{3}\-|[0-9]{3}){2}([0-9]{4}){1}$", searchstring)
    else:
        phoneMatch = re.search(r"^(\([0-9]{3}\)\s|[0-9]{3}\-|[0-9]{3}){1}(([0-9]{4}))$",searchstring)

    return True if phoneMatch else False
        
def problem2(searchstring):
    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    address = re.search(r'(\d+\s)*([A-Z]\D+\s)([Ave.]{4}|[St.]{3}|[Rd.]{3}|[Dr.]{3})', searchstring)
    return address.group(1) + address.group(2).strip() if address else False
    
def problem3(searchstring):
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    streetAddr = re.search(r'((.*)(?P<street_num>[0-9]+\s)(?P<street_name>[\w\s]+)(?P<road>('r'Rd.|Dr.|Ave.|St.))([\w\s]*))', searchstring)

    streetName = streetAddr['street_name'].strip()
    reverseStreet = streetName[::-1]

    replaceStreet = re.sub(r'([0-9]+\s)'+streetName, r'\1'+reverseStreet, streetAddr.groups()[0])

    return replaceStreet


if __name__ == '__main__' :
    print("\nProblem 1:")
    print("Answer correct?", problem1('+1 765-494-4600') == True)
    print("Answer correct?", problem1('+52 765-494-4600 ') == False)
    print("Answer correct?", problem1('+1 (765) 494 4600') == False)
    print("Answer correct?", problem1('+52 (765) 494-4600') == True)
    print("Answer correct?", problem1('+52 7654944600') == True)
    print("Answer correct?", problem1('494-4600') == True)

    print("\nProblem 2:")
    print("Answer correct?",problem2('Please flip your wallet at 465 Northwestern Ave.') == "465 Northwestern")
    print("Answer correct?",problem2('Meet me at 201 South First St. at noon') == "201 South First")
    print("Answer correct?",problem2('Type "404 Not Found St" on your phone at 201 South First St. at noon') == "201 South First")
    print("Answer correct?",problem2("123 Mayb3 Y0u 222 Did not th1nk 333 This Through Rd. Did Y0u Ave.") == "333 This Through")
    print("\nProblem 3:")
    print("Answer correct?",problem3('The EE building is at 465 Northwestern Ave.') == "The EE building is at 465 nretsewhtroN Ave.")
    print("Answer correct?",problem3('Meet me at 201 South First St. at noon') == "Meet me at 201 tsriF htuoS St. at noon")
