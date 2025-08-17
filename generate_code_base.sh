#!/bin/bash


# Define the output file
OUTPUT_FILE="code_base.txt"
# Temporary file to hold all concatenated content before filtering
TEMP_ALL_CONTENTS="temp_all_contents.txt"
# Temporary file for the sed script
SED_SCRIPT_FILE="filter_rules.sed"
# Define the script's own filename for exclusion
SCRIPT_FILENAME="generate_code_base.sh" # Assuming the script is named this. Adjust if necessary.
# Configuration file for additional user-defined exclusion patterns
EXCLUDE_CONFIG_FILE="exclude_patterns.conf"

export OUTPUT_FILE TEMP_ALL_CONTENTS SED_SCRIPT_FILE SCRIPT_FILENAME EXCLUDE_CONFIG_FILE

echo "--- Script Start ---"
echo "Script Filename: $SCRIPT_FILENAME"
echo "Output file: $OUTPUT_FILE"
echo "Temp content file: $TEMP_ALL_CONTENTS"
echo "Sed script file: $SED_SCRIPT_FILE"
echo "User Exclusions Config File: $EXCLUDE_CONFIG_FILE"

# --- Configuration: Define patterns to EXCLUDE ---
# Core essential exclusions that should always be present
EXCLUDE_PATTERNS=(
    ".git/"                 # Git repository data
    "$OUTPUT_FILE"          # The output file itself
    "$TEMP_ALL_CONTENTS"    # Temporary file for all content
    "$SED_SCRIPT_FILE"      # Temporary sed script
    "$SCRIPT_FILENAME"      # The script itself
    # "$EXCLUDE_CONFIG_FILE"  # The exclusion config file itself
)

# Read additional patterns from the user's config file, if it exists
if [ -f "$EXCLUDE_CONFIG_FILE" ]; then
    echo "Reading additional exclusion patterns from $EXCLUDE_CONFIG_FILE..."
    while IFS= read -r line || [ -n "$line" ]; do
        trimmed_line=$(echo "$line" | awk '{$1=$1};1') 
        if [ -n "$trimmed_line" ] && [[ ! "$trimmed_line" =~ ^# ]]; then
            EXCLUDE_PATTERNS+=("$trimmed_line")
        fi
    done < "$EXCLUDE_CONFIG_FILE"
    echo "Finished reading $EXCLUDE_CONFIG_FILE."
else
    echo "User exclusions config file ($EXCLUDE_CONFIG_FILE) not found. Using default exclusions only."
    echo "You can create $EXCLUDE_CONFIG_FILE and add one pattern per line to exclude more files/dirs."
fi

echo "Final EXCLUDE_PATTERNS defined:"
printf "%s\n" "${EXCLUDE_PATTERNS[@]}"


# Define patterns for files and directories to ignore for the 'tree' command.
TREE_IGNORE_ITEMS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    TREE_IGNORE_ITEMS+=("$(echo "$pattern" | sed 's:/*$::')") # Remove trailing slash for tree name match
done
TREE_IGNORE_PATTERN=$(IFS=\| ; echo "${TREE_IGNORE_ITEMS[*]}")
echo "Tree ignore pattern: $TREE_IGNORE_PATTERN"


# 1. Create/overwrite the output file with the directory tree structure.
echo "Generating directory tree (ignoring: $TREE_IGNORE_PATTERN)..."
tree -I "$TREE_IGNORE_PATTERN" > "$OUTPUT_FILE"
echo "Tree generation complete."

# 2. Add a separator to the output file.
echo -e "\n\n=======================================\n        FILE CONTENTS START HERE         \n=======================================\n" >> "$OUTPUT_FILE"
echo "Separator added to $OUTPUT_FILE."

# 3. Define a function to process each file found by find
process_file() {
    filepath="$1"
    # These variables are available due to 'export'
    # output_file_check="$OUTPUT_FILE" 
    # temp_contents_check="$TEMP_ALL_CONTENTS"
    # sed_script_check="$SED_SCRIPT_FILE"
    # this_script_name_check="$SCRIPT_FILENAME"
    # exclude_conf_check="$EXCLUDE_CONFIG_FILE"

    normalized_filepath="${filepath#./}"

    # Minimal Debugging (uncomment for more verbosity if needed)
    # echo "DEBUG PROCESS_FILE: Processing file: [$normalized_filepath]"

    # Check against core script-related files. Directory exclusions are handled by find -prune.
    if [ "$normalized_filepath" = "$OUTPUT_FILE" ] || \
       [ "$normalized_filepath" = "$TEMP_ALL_CONTENTS" ] || \
       [ "$normalized_filepath" = "$SED_SCRIPT_FILE" ] || \
    #    [ "$normalized_filepath" = "$SCRIPT_FILENAME" ] || \
       [ "$normalized_filepath" = "$SCRIPT_FILENAME" ]; then

    #    [ "$normalized_filepath" = "$EXCLUDE_CONFIG_FILE" ]; then
        # echo "DEBUG PROCESS_FILE: SKIPPING (core exclusion) file: [$normalized_filepath]"
        return 0 # Skip this file
    fi
    
    echo "--- START OF FILE: $normalized_filepath ---" >> "$TEMP_ALL_CONTENTS"
    cat "$filepath" >> "$TEMP_ALL_CONTENTS" 2>&1 # Capture stderr from cat
    if [ $? -ne 0 ]; then
        echo "DEBUG PROCESS_FILE: WARNING - cat command failed for file: [$filepath] with exit code $?. Stderr from cat should be visible above."
    fi
    echo -e "\n--- END OF FILE: $normalized_filepath ---" >> "$TEMP_ALL_CONTENTS"
}
export -f process_file # Export the function so xargs subshells can find it

# Concatenate file contents with markers into a temporary file.
echo "Gathering file contents into $TEMP_ALL_CONTENTS (using find -prune and xargs)..."
rm -f "$TEMP_ALL_CONTENTS" # Ensure temp file is clean

# --- Construct the find command with -prune for directories ---
find_command_args=("find" ".")
has_prune_conditions=false

for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$pattern" == */ && "$pattern" != "./" ]]; then # It's a directory pattern, not just "./"
        dir_name_for_path="${pattern%/}" # Remove trailing slash
        if [ -n "$dir_name_for_path" ]; then
            path_arg_for_find=""
            if [[ "$dir_name_for_path" == "."* ]]; then 
                 path_arg_for_find="./$dir_name_for_path"
            else 
                 path_arg_for_find="./$dir_name_for_path"
            fi
            
            find_command_args+=("-path" "$path_arg_for_find" "-prune" "-o")
            has_prune_conditions=true
        fi
    fi
done

# Append the action part for files not pruned: -type f -print0
find_command_args+=("-type" "f" "-print0")

echo "Executing find command and piping to xargs: ${find_command_args[*]}" 
"${find_command_args[@]}" | xargs -0 -I {} bash -c 'process_file "{}"'
find_xargs_exit_code=$? 
echo "Find and xargs process finished with exit code: $find_xargs_exit_code"


echo "Finished gathering file contents." 

if [ ! -f "$TEMP_ALL_CONTENTS" ]; then
    echo "CRITICAL ERROR: Temporary content file '$TEMP_ALL_CONTENTS' was NOT CREATED."
elif [ ! -s "$TEMP_ALL_CONTENTS" ]; then
    echo "CRITICAL WARNING: Temporary content file '$TEMP_ALL_CONTENTS' IS EMPTY."
else
    echo "$TEMP_ALL_CONTENTS was created and has size: $(wc -c < "$TEMP_ALL_CONTENTS") bytes."
fi

# 4. Filter the temporary file to exclude unwanted content and append to OUTPUT_FILE.
echo "Preparing to filter file contents from $TEMP_ALL_CONTENTS..."
rm -f "$SED_SCRIPT_FILE"
echo "# Sed script to remove excluded file blocks" > "$SED_SCRIPT_FILE"

for pattern_to_exclude in "${EXCLUDE_PATTERNS[@]}"; do
    escaped_sed_pattern=$(printf '%s\n' "$pattern_to_exclude" | sed 's:[][\/.^$*?+|()]:\\&:g')
    if [[ "$pattern_to_exclude" == */ ]]; then 
        echo "/^--- START OF FILE: ${escaped_sed_pattern}.* ---$/,/^--- END OF FILE: ${escaped_sed_pattern}.* ---$/d" >> "$SED_SCRIPT_FILE"
    else 
        echo "/^--- START OF FILE: ${escaped_sed_pattern} ---$/,/^--- END OF FILE: ${escaped_sed_pattern} ---$/d" >> "$SED_SCRIPT_FILE"
    fi
done

if [ -s "$SED_SCRIPT_FILE" ]; then
    echo "Generated $SED_SCRIPT_FILE. Lines: $(wc -l < "$SED_SCRIPT_FILE")"
else
    echo "Warning: $SED_SCRIPT_FILE is empty or was not created. No sed filtering rules generated."
fi

echo "Checking conditions for filtering..."
if [ -s "$SED_SCRIPT_FILE" ] && [ -s "$TEMP_ALL_CONTENTS" ]; then
    echo "Both $SED_SCRIPT_FILE and $TEMP_ALL_CONTENTS have content. Proceeding with sed."
    sed -f "$SED_SCRIPT_FILE" "$TEMP_ALL_CONTENTS" >> "$OUTPUT_FILE"
    echo "Filtering complete. Content appended to $OUTPUT_FILE."
elif [ ! -s "$TEMP_ALL_CONTENTS" ]; then
    echo "Skipping sed filtering because $TEMP_ALL_CONTENTS is empty or does not exist."
else
    echo "Skipping sed filtering because $SED_SCRIPT_FILE is empty (no rules), though $TEMP_ALL_CONTENTS might have content."
fi

# 5. Cleanup temporary files
echo "Cleaning up temporary files..."
rm -f "$TEMP_ALL_CONTENTS"
rm -f "$SED_SCRIPT_FILE"
echo "Temporary files removed."

echo "--- Script End ---"
echo "File mapping complete. Check $OUTPUT_FILE"
