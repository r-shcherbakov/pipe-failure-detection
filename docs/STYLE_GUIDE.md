# Personal Style Guides

## Git

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
and [Commitizen](https://github.com/commitizen-tools/commitizen)

The [Changelog](https://keepachangelog.com/en/1.0.0/) is an important part of a project (built with `commitizen`).
Use [SemVer](https://semver.org/)

### Conventional Commits

-   `type(scope): description / body`
-   The type feat MUST be used when a commit adds a new feature to your application or library.
-   The type fix MUST be used when a commit represents a bugfix for your application.
-   A scope MUST consist of a noun describing a section of the codebase surrounded by parenthesis,
    e.g., `fix(parser):` or issue number `fix(#32):`
-   A `!` can be used to indicate a breaking change, e.g. `refactor!: drop support for Node 6`
-   What if a commit fits multiple types?
    -   Go back and make multiple commits whenever possible.
        Part of the benefit of Conventional Commits is its ability to drive us to make more organized commits and PRs.
    -   It discourages moving fast in a disorganized way.
        It helps you be able to move fast long term across multiple projects with varied contributors.
-   `SemVer`: `fix : PATCH / feat : MINOR / BREAKING CHANGE : MAJOR`
    -   Use `git rebase -i` to fix commit names prior to merging if incorrect types/scopes are used

### Commitizen Types and Scopes

-   Types
    -   fix: A bugfix
    -   feat: A new feature
    -   docs: Documentation-only changes (code comments, separate docs)
    -   style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons)
    -   perf: A code change that improves performance
    -   refactor: A change to production code that is not fix, feat, or perf
    -   test: Adding missing or correcting existing tests
    -   build: Changes that affect the build tool or external dependencies (example scopes: pip, docker, npm)
    -   ci: Changes to our CI configuration files and scripts (example scopes: GitLabCI)
-   Scopes
    -   Class, File name, Issue Number, other approved noun

### Git Message Guidelines

-   [Commit message guidelines](https://writingfordevelopers.substack.com/p/how-to-write-a-commit-message)
    -   Full sentence with verb (_lowercase_) and concise description. Below are modified examples for Conventional Commits
        -   `fix(roles): bug in admin role permissions`
        -   `feat(ui): implement new button design`
        -   `build(pip): upgrade package to remove vulnerabilities`
        -   `refactor: file structure to improve code readability`
        -   `perf(cli): rewrite methods`
        -   `feat(api): endpoints to implement new customer dashboard`
-   [How to write a good commit message](https://chris.beams.io/posts/git-commit/)
    -   A diff will tell you what changed, but only the commit message can properly tell you why.
    -   Keep in mind: [This](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html)
        [has](https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines)
        [all](https://github.com/torvalds/subsurface-for-dirk/blob/master/README.md#contributing)
        [been](http://who-t.blogspot.co.at/2009/12/on-commit-messages.html)
        [said](https://github.com/erlang/otp/wiki/writing-good-commit-messages)
        [before](https://github.com/spring-projects/spring-framework/blob/30bce7/CONTRIBUTING.md#format-commit-messages).
    -   The seven rules of a great Git commit message
        -   [Try for 50 characters, but consider 72 the hard limit](https://chris.beams.io/posts/git-commit/#limit-50)
        -   [Use the body to explain what and why vs. how](https://chris.beams.io/posts/git-commit/#why-not-how)

### Issue Labels and Milestones

Personal Guide

-   Labels
    -   `Needs Discussion`: (#ff5722) Ticket needs discussion and prioritization
    -   `Type: Bug`: (#d73a4a) Something isn't working
    -   `Type: Documentation`: (#69cde9) Documentation changes
    -   `Type: Maintenance`: (#c5def5) Chore including build/dep, CI, refactor, or perf
    -   `Type: Idea`: (#fbca04) General idea or concept that could become a feature request
    -   `Type: Feature`: (#0075ca) Clearly defined new feature request
-   Milestones
    -   Current Tasks (Main Milestone) - name could change based on a specific project, sprint, or month
    -   Next Tasks
    -   Blue Sky
-   Search
    -   `is:open is:issue assignee:KyleKing archived:false milestone:"blue sky"` or `no:milestone` etc.