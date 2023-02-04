import pickle

def savevar(var):

    # Open a file and use dump()
    with open('savevars.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(var, file)

def loadvar():

    # Open the file in binary mode
    with open('savevars.pkl', 'rb') as file:
        # Call load method to deserialze
        var = pickle.load(file)
        print(var + 'mode selected')
        return var

def savevardet(var2):

    # Open a file and use dump()
    with open('savevardet.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(var2, file)

def loadvardet():

    # Open the file in binary mode
    with open('savevardet.pkl', 'rb') as file:
        # Call load method to deserialze
        var2 = pickle.load(file)
        print(var2 + 'mode selected')
        return var2

