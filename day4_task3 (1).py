# Step 1: Create sets for each friend's interests
friend_a = {"Python", "Cooking", "Hiking", "Movies"}
friend_b = {"Hiking", "Gaming", "Photography", "Python"}

# Step 2: Intersection (common interests)
shared_interests = friend_a & friend_b

# Step 3: Union (all unique interests)
all_interests = friend_a | friend_b

# Step 4: Difference (interests only friend_a has)
unique_to_a = friend_a - friend_b

# Step 5: Output results
print("Shared interests:", shared_interests)
print("All interests:", all_interests)
print("Interests only friend_a has:", unique_to_a)
