---
layout: page
title: "git cheat sheet"
description: "A list of solutions to common git problems"
tags: git
---
{% include JB/setup %}

## Misc

Hide untracked files

	git config status.showuntrackedfiles no # (guess how you undo this?)


## git reset

Unstage a file (no file argument resets all file).

	git reset <file> 

Discard changes and reset tree to a particular commit:

    git reset --hard <hash>


## git diff

Show changes between the Working Tree and the Index

    git diff

Show differences between the last commit and the Index (i.e. staged changes)

    git diff --cached

Shows the differences between the Working Tree and a specific commit

    git diff <hash>


## git show

Show the contents of a file at a particular commit

    git show <hash>:<file>


## git checkout

Apply changes from a single file to your working tree

    git checkout <hash> -- <file>

## git cherry-pick

Apply all changes from a particular commit without creating a new commit

    git cherry-pick --no-commit <hash>


## git stash

Apply a particular stash

	git stash apply stash@{<index>}   # to apply but keep the stash stored
	git stash pop stash@{<index>}     # to apply and remove stash

Apply a single file from stash

	git checkout stash@{<index>} -- <filename>		

Unapply a stash (assuming the previously applied stash hasn't been dropped)

	git stash show -p stash@{<index>} | git apply -R 

Stash only a select group of files

	git add <filename>      # add the things you *don't* want to stash
	git stash --keep-index
	git reset HEAD^         # unstage modified files



