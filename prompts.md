# My AI Development Command Palette

## @NewSession
Utilizing first principles, I want you to analyze all the markdown files in this project. Once you have a full understanding, please tell me what you know about the project's vision, principles, and current state in detail.

## @NewSessionProject
Utilizing first principles, I want you to analyze this project in detail. Once you have a full understanding, please tell me what you know about the project's vision, principles, and current state in detail.

## @Plan
Utilizing first principles, I want you to come up with an ideal plan of implementation. Once you have the plan, I want you to ask yourself why it won’t work. Then use the answer from that question to further redevelop and/or refine your plan. Do not generate any code yet, just walk me through your plan in detail.

## @PanelPlan
Utilizing first principles, I want you to generate three distinct implementation plans from the perspectives of:
A Senior Staff Engineer obsessed with scalability, long-term maintainability, and clean architecture.
A Pragmatic Startup CTO obsessed with speed, shipping a V1 MVP as fast as possible, and using off-the-shelf tools.
A Security Engineer obsessed with risk mitigation and real-time safety.
For each plan, list the pros, cons, and key trade-offs. Then, recommend which plan (or a hybrid of them) is the best fit and why.

##OverEngineering
From first principles, do you think we may be overengineering and overthinking what is actually much simpler? Why or why not?

## @PreMortem
This is a great plan. Now, I want you to perform a pre-mortem. Fast forward six weeks. The implementation of this plan has been a complete disaster. What was the single, non-obvious, and incorrect assumption we made in this plan that caused the failure? Based on that insight, what is the one change we must make to the plan *right now* to prevent that failure?

## @Implement
Please begin implementation of the approved plan.

## @PostMortem
Let’s pause. Now that we’re here, I want you to think about your mental space when you first started implementing this part of the project. What information do you wish you had when you started that would have saved you the most heartache and time? How would you instruct your past self or a new developer on how to attack this problem?

## @PostMortemInject
Before you begin, another developer started at this exact spot and spent a lot of time and energy trying to solve this issue. When said developer was asked what information did they wish they knew earlier that would have saved them the most time and heartache, they had the following to say. Please use their insights to come up with a first principles plan of implementation. From there ask yourself why won't this plan work? Then use the answer from that question to further redevelop and refine your plan. Do not generate any code just walk me through your plan in detail.

## @UpdateDocs
Please update and/or remove any relevant markdown files based on the changes we've implemented or insights we've developed. Generally it's better to update existing docs than creating new ones. Ensure that another developer could pick up from where we left off with all the context they need. Once done, please commit and push the changes to github.

## @CleanupDocs
I want you to utilize first principles to ensure that only necessary markdown files and markdown content within those files are in this project. I worry about documentation creep and want to make sure that only the minimum amount of content is available for a brand new developer with no context to pick up from where the project was last left off. Do not touch prompts.md

## @WeReady
I want you to think through first principles and try to answer the question, why might we not actually be ready for this implementation? What are the non obvious failure modes?
