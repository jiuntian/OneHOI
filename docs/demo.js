(function () {
    // Workflow 1: A Day at the Park
    const parkSteps = [
        {
            label: 'HOI Generation (Mixed Conditions)',
            prompt: 'A <span class="hl-red">girl</span> is walking a <span class="hl-red">dog</span> with a <span class="hl-red">leash</span>, while <span class="hl-green">another person</span> is sitting on a <span class="hl-green">bench</span> beside a <span class="hl-green">lamp post</span>',
            promptRaw: 'A girl is walking a dog with a leash, while another person is sitting on a bench beside a lamp post',
            mask: 'res/workflow_1/all_masks-1.png',
            sourceImage: null,
            outputImage: 'res/workflow_1/generated_girl.jpg',
            buttonText: 'Generate',
        },
        {
            label: 'HOI Editing (Both)',
            prompt: 'The <span class="hl-red">girl</span> is now holding the <span class="hl-red">dog</span>, while <span class="hl-green">another person</span> is lying on the <span class="hl-green">bench</span>',
            promptRaw: 'The girl is now holding the dog, while another person is lying on the bench',
            mask: 'res/workflow_1/all_masks.png',
            sourceImage: 'res/workflow_1/generated_girl.jpg',
            outputImage: 'res/workflow_1/edited_girl.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'HOI Editing (Single)',
            prompt: 'The <span class="hl-red">girl</span> is now holding a <span class="hl-red">ball</span>',
            promptRaw: 'The girl is now holding a ball',
            mask: 'res/workflow_1/all_masks_girl_ball_only_girl.png',
            sourceImage: 'res/workflow_1/edited_girl.jpg',
            outputImage: 'res/workflow_1/generated_edit_girl_hold_ball.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'Attribute Editing',
            prompt: 'Change the <span class="hl-red">lamp post</span> to <span class="hl-red">blue</span>',
            promptRaw: 'Change the lamp post to blue',
            mask: 'res/workflow_1/all_masks_girl_lamp.png',
            sourceImage: 'res/workflow_1/generated_edit_girl_hold_ball.jpg',
            outputImage: 'res/workflow_1/generated_edit_girl_blue_lamp.jpg',
            buttonText: 'Edit',
        },
    ];

    // Workflow 2: Lunar Exploration
    const lunarSteps = [
        {
            label: 'HOI Generation',
            prompt: 'An <span class="hl-red">astronaut</span> holds a <span class="hl-red">ONEHOI flag</span> on the lunar surface',
            promptRaw: 'An astronaut holds a ONEHOI flag on the lunar surface',
            mask: 'res/workflow_2/all_masks_gen_astronaut.png',
            sourceImage: null,
            outputImage: 'res/workflow_2/generated_gen_astronaut_flag.jpg',
            buttonText: 'Generate',
        },
        {
            label: 'HOI Editing (Layout-free)',
            prompt: 'The astronaut is now planting the flag into the lunar soil',
            promptRaw: 'The astronaut is now planting the flag into the lunar soil',
            mask: null,
            sourceImage: 'res/workflow_2/generated_gen_astronaut_flag.jpg',
            outputImage: 'res/workflow_2/generated_edit_astronaut_plant_flag.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'HOI Editing (Add)',
            prompt: 'Another <span class="hl-red">astronaut</span> is driving a <span class="hl-red">lunar rover</span>',
            promptRaw: 'Another astronaut is driving a lunar rover',
            mask: 'res/workflow_2/all_masks_rover.png',
            sourceImage: 'res/workflow_2/generated_edit_astronaut_plant_flag.jpg',
            outputImage: 'res/workflow_2/generated_edit_add_rover.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'Attribute Editing (Object Change)',
            prompt: 'Change the <span class="hl-red">Earth</span> to <span class="hl-red">Mars</span>',
            promptRaw: 'Change the Earth to Mars',
            mask: 'res/workflow_2/all_masks_earth.png',
            sourceImage: 'res/workflow_2/generated_edit_add_rover.jpg',
            outputImage: 'res/workflow_2/generated_edit_mars.jpg',
            buttonText: 'Edit',
        },
    ];

    // Workflow 3: Life of Pi
    const piSteps = [
        {
            label: 'HOI Generation (Mixed Conditions)',
            prompt: 'A <span class="hl-red">person</span> is standing on a <span class="hl-red">boat</span> with <span class="hl-green">floating debris</span> nearby',
            promptRaw: 'A person is standing on a boat with floating debris nearby',
            mask: 'res/workflow_life_of_pi/all_masks_raft.png',
            sourceImage: null,
            outputImage: 'res/workflow_life_of_pi/generated_gen_raft.jpg',
            buttonText: 'Generate',
        },
        {
            label: 'HOI Editing (Layout-guided)',
            prompt: 'The <span class="hl-red">person</span> is now paddling the <span class="hl-red">boat</span>',
            promptRaw: 'The person is now paddling the boat',
            mask: 'res/workflow_life_of_pi/all_masks_raft_paddle.png',
            sourceImage: 'res/workflow_life_of_pi/generated_gen_raft.jpg',
            outputImage: 'res/workflow_life_of_pi/generated_edit_person_paddle_boat.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'HOI Editing (Add)',
            prompt: 'Add a <span class="hl-red">white Bengal tiger</span> roaring and lying on the <span class="hl-red">boat</span>',
            promptRaw: 'Add a white Bengal tiger roaring and lying on the boat',
            mask: 'res/workflow_life_of_pi/all_masks_raft_tiger2.png',
            sourceImage: 'res/workflow_life_of_pi/generated_edit_person_paddle_boat.jpg',
            outputImage: 'res/workflow_life_of_pi/generated_edit_add_tiger2.jpg',
            buttonText: 'Edit',
        },
        {
            label: 'Attribute Editing (Scene)',
            prompt: 'Change the scene to <span class="hl-red">stormy turbulent</span>',
            promptRaw: 'Change the scene to stormy turbulent',
            mask: null,
            sourceImage: 'res/workflow_life_of_pi/generated_edit_add_tiger2.jpg',
            outputImage: 'res/workflow_life_of_pi/generated_edit_stormy3.jpg',
            buttonText: 'Edit',
        },
    ];

    const workflows = [
        { title: 'Lunar Exploration', steps: lunarSteps },
        { title: 'Life of Pi', steps: piSteps },
        { title: 'A Day at the Park', steps: parkSteps },
    ];

    let currentWorkflow = 0;
    let currentStep = 0;
    let isAnimating = false;
    let autoTimer = null;
    let typingTimer = null;

    // DOM elements
    const promptEl = document.getElementById('demo-prompt');
    const maskEl = document.getElementById('demo-mask');
    const sourceEl = document.getElementById('demo-source');
    const outputEl = document.getElementById('demo-output');
    const actionBtn = document.getElementById('demo-action-btn');
    const labelEl = document.getElementById('demo-step-label');
    const stepsNav = document.getElementById('demo-steps-nav');
    const sourceOverlay = document.getElementById('demo-source-overlay');
    const outputOverlay = document.getElementById('demo-output-overlay');
    const maskOverlay = document.getElementById('demo-mask-overlay');
    const workflowNav = document.getElementById('demo-workflow-nav');

    function getSteps() {
        return workflows[currentWorkflow].steps;
    }

    function clearTimers() {
        if (autoTimer) { clearTimeout(autoTimer); autoTimer = null; }
        if (typingTimer) { clearTimeout(typingTimer); typingTimer = null; }
    }

    function typeText(el, html, raw, callback) {
        el.innerHTML = '';
        el.classList.add('typing');
        let i = 0;
        const speed = 35;
        function type() {
            if (i < raw.length) {
                el.textContent = raw.substring(0, i + 1);
                i++;
                typingTimer = setTimeout(type, speed);
            } else {
                el.innerHTML = html;
                el.classList.remove('typing');
                if (callback) callback();
            }
        }
        type();
    }

    function fadeInMask(step, callback) {
        if (step.mask) {
            maskEl.src = step.mask;
            maskEl.style.display = 'block';
            maskOverlay.style.display = 'none';
            maskEl.style.opacity = '0';
            requestAnimationFrame(() => {
                maskEl.style.transition = 'opacity 0.8s ease';
                maskEl.style.opacity = '1';
                setTimeout(() => {
                    if (callback) callback();
                }, 900);
            });
        } else {
            maskEl.style.display = 'none';
            maskOverlay.style.display = 'flex';
            if (callback) callback();
        }
    }

    function showSource(step) {
        if (step.sourceImage) {
            sourceEl.src = step.sourceImage;
            sourceEl.style.display = 'block';
            sourceOverlay.style.display = 'none';
        } else {
            sourceEl.style.display = 'none';
            sourceOverlay.style.display = 'flex';
        }
    }

    function animateStep(stepIdx) {
        if (isAnimating) return;
        isAnimating = true;
        clearTimers();

        const steps = getSteps();
        currentStep = stepIdx;
        const step = steps[stepIdx];

        updateStepNav();
        labelEl.textContent = step.label;

        // Reset output
        outputEl.style.display = 'none';
        outputOverlay.style.display = 'flex';

        // Reset action button
        actionBtn.textContent = step.buttonText;
        actionBtn.classList.remove('clicked', 'pulse', 'done');
        actionBtn.disabled = true;

        showSource(step);

        // Reset mask
        maskEl.style.display = 'none';
        maskOverlay.style.display = 'flex';

        // Typing animation
        typeText(promptEl, step.prompt, step.promptRaw, () => {
            fadeInMask(step, () => {
                actionBtn.disabled = false;
                actionBtn.classList.add('pulse');

                autoTimer = setTimeout(() => {
                    triggerGenerate();
                }, 2500);
            });
        });
    }

    function triggerGenerate() {
        clearTimers();
        const steps = getSteps();
        const step = steps[currentStep];
        actionBtn.classList.remove('pulse');
        actionBtn.classList.add('clicked');
        actionBtn.disabled = true;

        actionBtn.textContent = '⏳ Processing...';

        setTimeout(() => {
            outputEl.src = step.outputImage;
            outputEl.style.display = 'block';
            outputOverlay.style.display = 'none';
            outputEl.style.opacity = '0';
            requestAnimationFrame(() => {
                outputEl.style.transition = 'opacity 0.6s ease';
                outputEl.style.opacity = '1';
            });

            actionBtn.textContent = '✓ Done!';
            actionBtn.classList.remove('clicked');
            actionBtn.classList.add('done');

            isAnimating = false;

            autoTimer = setTimeout(() => {
                actionBtn.classList.remove('done');
                const next = currentStep + 1;
                if (next < steps.length) {
                    animateStep(next);
                } else {
                    // Move to next workflow
                    const nextWorkflow = (currentWorkflow + 1) % workflows.length;
                    switchWorkflow(nextWorkflow);
                }
            }, 3000);
        }, 1000);
    }

    function updateStepNav() {
        const dots = stepsNav.querySelectorAll('.demo-step-dot');
        dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === currentStep);
        });
    }

    function updateWorkflowNav() {
        const tabs = workflowNav.querySelectorAll('.demo-workflow-tab');
        tabs.forEach((tab, i) => {
            tab.classList.toggle('active', i === currentWorkflow);
        });
    }

    function buildStepNav() {
        const steps = getSteps();
        stepsNav.innerHTML = '';
        steps.forEach((step, i) => {
            const dot = document.createElement('button');
            dot.className = 'demo-step-dot' + (i === 0 ? ' active' : '');
            dot.textContent = (i + 1);
            dot.title = step.label;
            dot.addEventListener('click', () => {
                if (isAnimating && currentStep !== i) {
                    isAnimating = false;
                    clearTimers();
                }
                animateStep(i);
            });
            stepsNav.appendChild(dot);
        });
    }

    function buildWorkflowNav() {
        workflowNav.innerHTML = '';
        workflows.forEach((wf, i) => {
            const tab = document.createElement('button');
            tab.className = 'demo-workflow-tab' + (i === 0 ? ' active' : '');
            tab.textContent = wf.title;
            tab.addEventListener('click', () => {
                if (i !== currentWorkflow) {
                    isAnimating = false;
                    clearTimers();
                    switchWorkflow(i);
                }
            });
            workflowNav.appendChild(tab);
        });
    }

    function switchWorkflow(idx) {
        currentWorkflow = idx;
        currentStep = 0;
        isAnimating = false;
        clearTimers();
        updateWorkflowNav();
        buildStepNav();
        animateStep(0);
    }

    // Action button click
    actionBtn.addEventListener('click', () => {
        if (!actionBtn.disabled) {
            triggerGenerate();
        }
    });

    // Init
    buildWorkflowNav();
    buildStepNav();
    animateStep(0);
})();
