# Function to calculate area and perimeter of a rectangle
def calc_rectangle(length, width):
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter  # returning two values as a tuple

# Take user input
length = float(input("Enter the length: "))
width = float(input("Enter the width: "))

# Call the function
area, perimeter = calc_rectangle(length, width)

# Print results
print(f"Area: {area}, Perimeter: {perimeter}")
