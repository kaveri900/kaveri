# Step 1: Raw login data with duplicates
raw_logs = ["ID01", "ID02", "ID01", "ID05", "ID02", "ID08", "ID01"]

# Step 2: Convert list to set to remove duplicates
unique_users = set(raw_logs)

# Step 3: Membership test
is_ID05_present = "ID05" in unique_users

# Step 4: Compare counts
original_count = len(raw_logs)
unique_count = len(unique_users)

# Step 5: Output results
print("Unique Users:", unique_users)
print("Is ID05 present?", is_ID05_present)
print("Original log count:", original_count)
print("Unique user count:", unique_count)
