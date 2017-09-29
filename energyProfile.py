#! /usr/bin/env python

def isnumeric(string):
    """Checks if the given string represents a floating number or integer
    """
    string = string.replace('-', '', 1) # Remove sign
    string = string.replace('.', '', 1) # Remove dot
    
    return string.isdigit() # Check if the result is a positive integer



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import sys
    
    # Read all lines on STDIN
    lines = sys.stdin.readlines()

    # Get the energy profile values
    energyProfile = [ int(e) for e in lines.pop().strip().split(' ') if e.replace('-', '', 1).isdigit() ]
    print("Energy values: ", energyProfile)

    # Output the sequence
    sequence = ''.join(lines)
    print(sequence)

    # Plot the energy profile
    plt.plot(energyProfile)
    plt.ylabel('Energy profile')
    plt.show()
