# edX

Compilation of code used for projects or homework within the edX space, including coursework from programs such as the GTx MicroMasters.

---

## Repository Structure

This repository is organized by course.  
Each course lives in its own top-level folder using the following naming convention:

```
<InstitutionAbbrev>.<CourseNumber>
```

Examples:
- `GTx.ISYE6501` — Georgia Tech
- `UTx.CS303E` — University of Texas (example)

Each course folder follows a standard internal structure:
```
homework/
projects/
data/
notes/
```

---

## Initial Setup (One-Time)

Clone the repository:
```bash
git clone https://github.com/<your-username>/edX.git
cd edX
```

---

## Adding a New Course

1. Choose a course identifier using the format:
```
<InstitutionAbbrev>.<CourseNumber>
```

2. Create the course folder and standard subfolders:
```bash
mkdir -p <COURSE_ID>/{homework,projects,data,notes}
```

3. Add placeholder files so Git tracks the folders:
```bash
touch <COURSE_ID>/README.md
touch <COURSE_ID>/{homework,projects,data,notes}/.gitkeep
```

4. Commit the changes:
```bash
git add .
git commit -m "Add course <COURSE_ID>"
git push
```
