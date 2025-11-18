<!-- ---
!-- Timestamp: 2025-10-29 10:42:52
!-- Author: ywatanabe
!-- File: /home/ywatanabe/.claude/commands/git.md
!-- --- -->

1. Understand the version control guideline.

2. Based on the guideline, conduct version control on the current codebase using `git` and `gh`.

3. Do not push sensitive data

4. Report what commands did you execute for this request in a simple with bullet point.
  - e.g.
    ```plaintext
    === Executed Git/GitHub Commands ===
    1. `git status`
    2. `git diff`
    3. `git log --oneline -5`
    4. `git add xxx/yyy.ext`
    5. `git add xxx/zzz.ext`
    6. `git commit -m "COMMIT MESSAGE"`
    7. `git status`
    ```

5. Show the raw contents of `git status`
    ```plaintext
    === git status ===
    `GIT STATUS` CONTENTS HERE
    ```
6. Report potential unexpected or unintended changes for the user
   ```plaintext
   === Potential unexpected changes ===
   None
   ```
   or
   ```plaintext
   === Potential unexpected changes ===
   XXX would be destructive and you may want to consider checking YYY.
   ```

7. Suggest next available steps with allocating numbers to select in the order of your recommendation:
   (Please ensure add an empty line between plans for visualization purposes)
   ``` plaintext
   === Available Plans ===
   0. Plan All:
      Try to execute all the plans below in the order. 
      If problem found, suggest next available plans again in the same manner.

   1. Plan A: **Merge back into develop**
     `git switch develop`
     `git merge feature/xxx`
  
   2. Plan B: **Push to orign**
     `git push develop origin/develop`
  
   3. Plan C: **Create PR from `origin/develop` to `origin/main`**
     `gh pr create --base main --head develop --title "PR MESSAGE" --body "PR BODY"`

   4. Plan D: **Merge the PR**
     `gh pr merge --auto --merge

   6. Plan X: **Revert to the last stable commit**
     `Since the changes disrupted important functionality XXX and reverting will be more reliable and straightforward, I recommend to checkout to `COMMIT HASH: COMMIT MESSAGE``
   ```

<!-- EOF -->