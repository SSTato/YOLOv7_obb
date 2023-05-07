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
        print(var + ' loss mode selected')
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
        print(var2 + ' notebook mode selected')
        return var2

def savevarang(var3):

    # Open a file and use dump()
    with open('savevarang.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(var3, file)

def loadvarang():

    # Open the file in binary mode
    with open('savevarang.pkl', 'rb') as file:
        # Call load method to deserialze
        var3 = pickle.load(file)
        print(var3 + ' angular definition mode selected')
        return var3


def savename(var4):
    
    # Open a file and use dump()
    with open('savename.pkl', 'wb') as file:
        # A new file will be created
        pickle.dump(var4, file)


def loadname():

    # Open the file in binary mode
    with open('savename.pkl', 'rb') as file:
        # Call load method to deserialze
        var4 = pickle.load(file)
        print(var4 + ' folder name selected')
        return var4
