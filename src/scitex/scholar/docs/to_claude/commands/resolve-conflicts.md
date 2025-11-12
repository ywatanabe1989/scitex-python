<!-- ---
!-- Timestamp: 2025-05-10 09:23:41
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.claude/commands/resolve-conflicts.md
!-- --- -->

Please resole merge conflicts. You might want to use the following scripts but please check they will not introduce destructive results.

## Rule-based Resolvers for ours/yours/newer

``` bash
#!/bin/bash
# Accepts arguments: --ours, --theirs, or --newer-timestamp

mode="$1"
if [[ -z "$mode" ]]; then
    echo "Usage: $0 [--ours|--theirs|--newer-timestamp]"
    exit 1
fi

# Find files with conflicts
git diff --name-only --diff-filter=U | while read file; do
    case "$mode" in
        --ours)
            git checkout --ours "$file"
            ;;
        --theirs)
            git checkout --theirs "$file"
            ;;
        --newer-timestamp)
            # For timestamp conflicts, this requires more complex parsing
            # Use the elisp function instead for Emacs files
            ;;
    esac
    git add "$file"
    echo "Resolved: $file"
done

echo "All conflicts resolved according to $mode strategy"
```

## Timestamp Resolver For Elisp Scripts
``` bash
#!/bin/bash
# Script to automatically resolve Git merge conflicts in timestamp lines

# Find files with timestamp merge conflicts
find . -type f -not -path "*/\.git/*" -exec grep -l "<<<<<<< HEAD" {} \; | while read file_path; do
    # Check if this file has timestamp conflicts
    if grep -q "^<<<<<<< HEAD\n.*Timestamp.*\n=======\n.*Timestamp.*\n>>>>>" "$file_path"; then
        echo "Processing timestamp conflicts in: $file_path"
        
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Process the file line by line
        in_conflict=false
        timestamp_head=""
        timestamp_branch=""
        
        while IFS= read -r line; do
            if [[ "$line" == "<<<<<<< HEAD" ]]; then
                in_conflict=true
                continue
            elif [[ "$in_conflict" == true && "$line" =~ "Timestamp:" ]]; then
                timestamp_head="$line"
                continue
            elif [[ "$line" == "=======" ]]; then
                if [[ "$in_conflict" == true ]]; then
                    continue
                else
                    echo "$line" >> "$temp_file"
                fi
            elif [[ "$in_conflict" == true && "$line" =~ "Timestamp:" ]]; then
                timestamp_branch="$line"
                continue
            elif [[ "$line" =~ ">>>>>" ]]; then
                if [[ "$in_conflict" == true ]]; then
                    # Extract and compare timestamps
                    ts_head=$(echo "$timestamp_head" | grep -oE "<[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}>")
                    ts_branch=$(echo "$timestamp_branch" | grep -oE "<[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}>")
                    
                    # Compare timestamps and use the newer one
                    if [[ "$ts_head" > "$ts_branch" ]]; then
                        echo "$timestamp_head" >> "$temp_file"
                    else
                        echo "$timestamp_branch" >> "$temp_file"
                    fi
                    
                    in_conflict=false
                    continue
                else
                    echo "$line" >> "$temp_file"
                fi
            elif [[ "$in_conflict" == false ]]; then
                echo "$line" >> "$temp_file"
            fi
        done < "$file_path"
        
        # Replace original file with the processed one
        mv "$temp_file" "$file_path"
        echo "Resolved timestamp conflicts in: $file_path"
    fi
done

echo "Timestamp conflict resolution completed"
```

<!-- EOF -->