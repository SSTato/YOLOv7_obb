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
