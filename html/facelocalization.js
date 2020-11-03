/************************* 
 * Facelocalization Test *
 *************************/

import { PsychoJS } from './lib/core-2020.1.js';
import * as core from './lib/core-2020.1.js';
import { TrialHandler } from './lib/data-2020.1.js';
import { Scheduler } from './lib/util-2020.1.js';
import * as util from './lib/util-2020.1.js';
import * as visual from './lib/visual-2020.1.js';
import * as sound from './lib/sound-2020.1.js';

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0, 0, 0]),
  units: 'height',
  waitBlanking: true
});

// store info about the experiment session:
let expName = 'facelocalization';  // from the Builder filename that created this script
let expInfo = {'participant': '', 'Student ID number (this is for RPP purpose only)': '', 'Age': '', 'Gender': '', 'Handedness': '', 'In this experiment, you need to constantly adjust the position of your cursor. what tool are your using to complete this experiment (mouse, trackpad,etc)?': '', 'Do you have normal or correct to normal vision?': ''};

// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(ConsentRoutineBegin());
flowScheduler.add(ConsentRoutineEachFrame());
flowScheduler.add(ConsentRoutineEnd());
flowScheduler.add(WelcomeRoutineBegin());
flowScheduler.add(WelcomeRoutineEachFrame());
flowScheduler.add(WelcomeRoutineEnd());
flowScheduler.add(IntroRoutineBegin());
flowScheduler.add(IntroRoutineEachFrame());
flowScheduler.add(IntroRoutineEnd());
flowScheduler.add(Intro_2RoutineBegin());
flowScheduler.add(Intro_2RoutineEachFrame());
flowScheduler.add(Intro_2RoutineEnd());
flowScheduler.add(Intro_3RoutineBegin());
flowScheduler.add(Intro_3RoutineEachFrame());
flowScheduler.add(Intro_3RoutineEnd());
flowScheduler.add(Intro_4RoutineBegin());
flowScheduler.add(Intro_4RoutineEachFrame());
flowScheduler.add(Intro_4RoutineEnd());
flowScheduler.add(CheckRoutineBegin());
flowScheduler.add(CheckRoutineEachFrame());
flowScheduler.add(CheckRoutineEnd());
flowScheduler.add(DemoRoutineBegin());
flowScheduler.add(DemoRoutineEachFrame());
flowScheduler.add(DemoRoutineEnd());
const demoLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(demoLoopBegin, demoLoopScheduler);
flowScheduler.add(demoLoopScheduler);
flowScheduler.add(demoLoopEnd);
flowScheduler.add(startRoutineBegin());
flowScheduler.add(startRoutineEachFrame());
flowScheduler.add(startRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin, trialsLoopScheduler);
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);
flowScheduler.add(EndRoutineBegin());
flowScheduler.add(EndRoutineEachFrame());
flowScheduler.add(EndRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  });


var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2020.1.2';
  expInfo['OS'] = window.navigator.platform;

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  
  return Scheduler.Event.NEXT;
}


var ConsentClock;
var consent;
var agree;
var disagree;
var mouse_5;
var WelcomeClock;
var text;
var key_resp_11;
var IntroClock;
var text_2;
var key_resp;
var Intro_2Clock;
var text_3;
var key_resp_2;
var Intro_3Clock;
var text_4;
var key_resp_3;
var Intro_4Clock;
var text_7;
var key_resp_10;
var CheckClock;
var text_5;
var key_resp_4;
var DemoClock;
var text_11;
var key_resp_8;
var exampleClock;
var image_2;
var mouse_2;
var image_5;
var backtocenterClock;
var image_4;
var mouse_4;
var startClock;
var text_8;
var key_resp_6;
var trialClock;
var image;
var mouse;
var image_6;
var Pause_1Clock;
var text_6;
var key_resp_5;
var EndClock;
var text_10;
var key_resp_7;
var globalClock;
var routineTimer;
function experimentInit() {
  // Initialize components for Routine "Consent"
  ConsentClock = new util.Clock();
  consent = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent', units : undefined, 
    image : 'consent.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [1.6, 0.9],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  agree = new visual.ImageStim({
    win : psychoJS.window,
    name : 'agree', units : undefined, 
    image : 'agree.png', mask : undefined,
    ori : 0, pos : [0.7, (- 0.45)], size : [0.15, 0.045],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  disagree = new visual.ImageStim({
    win : psychoJS.window,
    name : 'disagree', units : undefined, 
    image : 'disagree.png', mask : undefined,
    ori : 0, pos : [(- 0.7), (- 0.45)], size : [0.15, 0.045],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  mouse_5 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_5.mouseClock = new util.Clock();
  // Initialize components for Routine "Welcome"
  WelcomeClock = new util.Clock();
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: 'Welcome to the experiment. It is highly recommended that you complete it with a google chrome browser to minimize potential technical issues. \n\nMake sure to keep your screen clean. \n\nThe entire experiment will last for roughly one hour, with a break per every 5-7 minutes.\n\nPlease make sure not to exit full screen mode during the experiment to ensure maximum accuracy. Please also refrain from zooming in the experiment window. \n\nThe ideal eye-to-screen distance can be approximated by the length of your arm. \n\n(Press the spacebar to proceed)\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_11 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  //function fillArray(origArr, len) {
  //  var arr = [];
  //  for (var i = 0; i < len; i++) {
  //    for (var j = 0; j < origArr.length; j++) {
  //        arr.push(origArr[j]);
  //    }
  //  }
  //  return arr;
  //}
  //
  //function shuffleArray(array) {
  //    for (var i = array.length - 1; i > 0; i--) {
  //        var j = Math.floor(Math.random() * (i + 1));
  //        var temp = array[i];
  //        array[i] = array[j];
  //        array[j] = temp;
  //    }
  //}
  
  //upright-low-female: 1-4;
  //upright-low-male: 5-8;
  //upright-high-female: 9-12;
  //upright-high-male: 13-16;
  //invert-low-female: 17-20;
  //invert-low-male: 21-24;
  //invert-high-female: 25-28;
  //invert-high-male: 29-32;
  
  //var uniqCond = [1,2,3,4,5,6,7,8];
  //var rep = 2*48;
  //var allTrials = fillArray(uniqCond,rep);
  //shuffleArray(allTrials);
  
  window.trialCount = 0
  // Initialize components for Routine "Intro"
  IntroClock = new util.Clock();
  text_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_2',
    text: 'In this experiment, please keep your eyes fixed on the “+” at the center of your screen at ALL TIMES.\n\n(Press the spacebar  to proceed)\n\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Intro_2"
  Intro_2Clock = new util.Clock();
  text_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_3',
    text: 'You will see a circular target appearing on the screen very briefly. Magic letter: m\n\n(Press the spacebar to proceed)\n\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Intro_3"
  Intro_3Clock = new util.Clock();
  text_4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_4',
    text: 'Your task is to move your cursor to click the center of the target after it disappears. \n\nPlease try to remain fixated on the “+” while you adjust the cursor.\n\n(Press the spacebar to proceed)\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Intro_4"
  Intro_4Clock = new util.Clock();
  text_7 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_7',
    text: 'After clicking on the location of the target,  click on the “+” to start the next trial. \n\n(Press the spacebar to proceed)\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_10 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Check"
  CheckClock = new util.Clock();
  text_5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_5',
    text: 'Please enter the magic letter on the keyboard to continue.\nOtherwise, please reread the following instruction carefully to find the magic letter.\n \nIn this experiment, you need to fixate on the “+” at the center of your screen at ALL TIMES.\nYou will see a circular target appearing on the screen very briefly. Magic letter: m.\nYour task is to move your cursor to indicate the center of the target after it disappears.\nAfter clicking the location of the target, click on the “+” to start the next trial.\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Demo"
  DemoClock = new util.Clock();
  text_11 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_11',
    text: 'You can now see the experiment in action with a short demo.\n\nNote that in the actual experiment, the stimuli occurs for a much shorter duration than in this demo. \n\n(Press the spacebar to proceed)',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_8 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "example"
  exampleClock = new util.Clock();
  image_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_2', units : 'height', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.3, 0.3],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  mouse_2 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_2.mouseClock = new util.Clock();
  image_5 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_5', units : undefined, 
    image : 'fixation.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "backtocenter"
  backtocenterClock = new util.Clock();
  image_4 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_4', units : undefined, 
    image : 'target.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [0.1, 0.1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  mouse_4 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_4.mouseClock = new util.Clock();
  // Initialize components for Routine "start"
  startClock = new util.Clock();
  text_8 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_8',
    text: 'The experiment will now start, press the spacebar to proceed. \n\nPlease keep fixated ALL THE TIME. Please try to respond as soon as possible and DO NOT pause within experiment.\n\nPlease DON’T PRESS ESCAPE KEY or quit the full screen AT ANY TIME before you finish the experiment.\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_6 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image', units : 'height', 
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : [0.3, 0.3],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  mouse = new core.Mouse({
    win: psychoJS.window,
  });
  mouse.mouseClock = new util.Clock();
  image_6 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_6', units : undefined, 
    image : 'fixation.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -2.0 
  });
  var imageIndex;
  var imageName;
  
  // Initialize components for Routine "backtocenter"
  backtocenterClock = new util.Clock();
  image_4 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_4', units : undefined, 
    image : 'target.png', mask : undefined,
    ori : 0, pos : [0, 0], size : [0.1, 0.1],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : 0.0 
  });
  mouse_4 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_4.mouseClock = new util.Clock();
  // Initialize components for Routine "Pause_1"
  Pause_1Clock = new util.Clock();
  window.runCount = 0;
  text_6 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_6',
    text: RestMessage,
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: -1.0 
  });
  
  key_resp_5 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "End"
  EndClock = new util.Clock();
  text_10 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_10',
    text: 'This is the end of the experiment. Thank you for your participation! \n\nYou will receive your RPP credits (or payment) in 1-2 working days. If you have any questions or concerns, please contact the experimenter Zixuan Wang at zixuan@berkeley.edu.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0,
    color: new util.Color('white'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_7 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var gotValidClick;
var ConsentComponents;
function ConsentRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Consent'-------
    t = 0;
    ConsentClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_5
    mouse_5.clicked_name = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    ConsentComponents = [];
    ConsentComponents.push(consent);
    ConsentComponents.push(agree);
    ConsentComponents.push(disagree);
    ConsentComponents.push(mouse_5);
    
    for (const thisComponent of ConsentComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


var prevButtonState;
var continueRoutine;
function ConsentRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Consent'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = ConsentClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *consent* updates
    if (frameN >= 0.0 && consent.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent.tStart = t;  // (not accounting for frame time here)
      consent.frameNStart = frameN;  // exact frame index
      
      consent.setAutoDraw(true);
    }

    
    // *agree* updates
    if (frameN >= 0.0 && agree.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      agree.tStart = t;  // (not accounting for frame time here)
      agree.frameNStart = frameN;  // exact frame index
      
      agree.setAutoDraw(true);
    }

    
    // *disagree* updates
    if (frameN >= 0.0 && disagree.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      disagree.tStart = t;  // (not accounting for frame time here)
      disagree.frameNStart = frameN;  // exact frame index
      
      disagree.setAutoDraw(true);
    }

    // *mouse_5* updates
    if (frameN >= 0.0 && mouse_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_5.tStart = t;  // (not accounting for frame time here)
      mouse_5.frameNStart = frameN;  // exact frame index
      
      mouse_5.status = PsychoJS.Status.STARTED;
      mouse_5.mouseClock.reset();
      prevButtonState = mouse_5.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_5.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      let buttons = mouse_5.getPressed();
      if (!buttons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = buttons;
        if (buttons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          for (const obj of [agree]) {
            if (obj.contains(mouse_5)) {
              gotValidClick = true;
              mouse_5.clicked_name.push(obj.name)
            }
          }
          if (gotValidClick === true) { // abort routine on response
            continueRoutine = false;
          }
        }
      }
    }
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of ConsentComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function ConsentRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Consent'-------
    for (const thisComponent of ConsentComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // store data for thisExp (ExperimentHandler)
    // the Routine "Consent" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_11_allKeys;
var WelcomeComponents;
function WelcomeRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Welcome'-------
    t = 0;
    WelcomeClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_11.keys = undefined;
    key_resp_11.rt = undefined;
    _key_resp_11_allKeys = [];
    // keep track of which components have finished
    WelcomeComponents = [];
    WelcomeComponents.push(text);
    WelcomeComponents.push(key_resp_11);
    
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function WelcomeRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Welcome'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = WelcomeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text* updates
    if (frameN >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }

    
    // *key_resp_11* updates
    if (frameN >= 0.0 && key_resp_11.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_11.tStart = t;  // (not accounting for frame time here)
      key_resp_11.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_11.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_11.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_11.clearEvents(); });
    }

    if (key_resp_11.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_11.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_11_allKeys = _key_resp_11_allKeys.concat(theseKeys);
      if (_key_resp_11_allKeys.length > 0) {
        key_resp_11.keys = _key_resp_11_allKeys[_key_resp_11_allKeys.length - 1].name;  // just the last key pressed
        key_resp_11.rt = _key_resp_11_allKeys[_key_resp_11_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function WelcomeRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Welcome'-------
    for (const thisComponent of WelcomeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_allKeys;
var IntroComponents;
function IntroRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Intro'-------
    t = 0;
    IntroClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    // keep track of which components have finished
    IntroComponents = [];
    IntroComponents.push(text_2);
    IntroComponents.push(key_resp);
    
    for (const thisComponent of IntroComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function IntroRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Intro'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = IntroClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_2* updates
    if (frameN >= 0.0 && text_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_2.tStart = t;  // (not accounting for frame time here)
      text_2.frameNStart = frameN;  // exact frame index
      
      text_2.setAutoDraw(true);
    }

    
    // *key_resp* updates
    if (frameN >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }

    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of IntroComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function IntroRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Intro'-------
    for (const thisComponent of IntroComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_2_allKeys;
var Intro_2Components;
function Intro_2RoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Intro_2'-------
    t = 0;
    Intro_2Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_2.keys = undefined;
    key_resp_2.rt = undefined;
    _key_resp_2_allKeys = [];
    // keep track of which components have finished
    Intro_2Components = [];
    Intro_2Components.push(text_3);
    Intro_2Components.push(key_resp_2);
    
    for (const thisComponent of Intro_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function Intro_2RoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Intro_2'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = Intro_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_3* updates
    if (frameN >= 0.0 && text_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_3.tStart = t;  // (not accounting for frame time here)
      text_3.frameNStart = frameN;  // exact frame index
      
      text_3.setAutoDraw(true);
    }

    
    // *key_resp_2* updates
    if (frameN >= 0.0 && key_resp_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_2.tStart = t;  // (not accounting for frame time here)
      key_resp_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.clearEvents(); });
    }

    if (key_resp_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_2.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_2_allKeys = _key_resp_2_allKeys.concat(theseKeys);
      if (_key_resp_2_allKeys.length > 0) {
        key_resp_2.keys = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].name;  // just the last key pressed
        key_resp_2.rt = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Intro_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Intro_2RoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Intro_2'-------
    for (const thisComponent of Intro_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_3_allKeys;
var Intro_3Components;
function Intro_3RoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Intro_3'-------
    t = 0;
    Intro_3Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_3.keys = undefined;
    key_resp_3.rt = undefined;
    _key_resp_3_allKeys = [];
    // keep track of which components have finished
    Intro_3Components = [];
    Intro_3Components.push(text_4);
    Intro_3Components.push(key_resp_3);
    
    for (const thisComponent of Intro_3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function Intro_3RoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Intro_3'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = Intro_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_4* updates
    if (frameN >= 0.0 && text_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_4.tStart = t;  // (not accounting for frame time here)
      text_4.frameNStart = frameN;  // exact frame index
      
      text_4.setAutoDraw(true);
    }

    
    // *key_resp_3* updates
    if (frameN >= 0.0 && key_resp_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_3.tStart = t;  // (not accounting for frame time here)
      key_resp_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.clearEvents(); });
    }

    if (key_resp_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_3.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_3_allKeys = _key_resp_3_allKeys.concat(theseKeys);
      if (_key_resp_3_allKeys.length > 0) {
        key_resp_3.keys = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].name;  // just the last key pressed
        key_resp_3.rt = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Intro_3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Intro_3RoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Intro_3'-------
    for (const thisComponent of Intro_3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Intro_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_10_allKeys;
var Intro_4Components;
function Intro_4RoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Intro_4'-------
    t = 0;
    Intro_4Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_10.keys = undefined;
    key_resp_10.rt = undefined;
    _key_resp_10_allKeys = [];
    // keep track of which components have finished
    Intro_4Components = [];
    Intro_4Components.push(text_7);
    Intro_4Components.push(key_resp_10);
    
    for (const thisComponent of Intro_4Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function Intro_4RoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Intro_4'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = Intro_4Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_7* updates
    if (frameN >= 0.0 && text_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_7.tStart = t;  // (not accounting for frame time here)
      text_7.frameNStart = frameN;  // exact frame index
      
      text_7.setAutoDraw(true);
    }

    
    // *key_resp_10* updates
    if (frameN >= 0.0 && key_resp_10.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_10.tStart = t;  // (not accounting for frame time here)
      key_resp_10.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_10.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_10.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_10.clearEvents(); });
    }

    if (key_resp_10.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_10.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_10_allKeys = _key_resp_10_allKeys.concat(theseKeys);
      if (_key_resp_10_allKeys.length > 0) {
        key_resp_10.keys = _key_resp_10_allKeys[_key_resp_10_allKeys.length - 1].name;  // just the last key pressed
        key_resp_10.rt = _key_resp_10_allKeys[_key_resp_10_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Intro_4Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Intro_4RoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Intro_4'-------
    for (const thisComponent of Intro_4Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Intro_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_4_allKeys;
var CheckComponents;
function CheckRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Check'-------
    t = 0;
    CheckClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_4.keys = undefined;
    key_resp_4.rt = undefined;
    _key_resp_4_allKeys = [];
    // keep track of which components have finished
    CheckComponents = [];
    CheckComponents.push(text_5);
    CheckComponents.push(key_resp_4);
    
    for (const thisComponent of CheckComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function CheckRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Check'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = CheckClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_5* updates
    if (frameN >= 0.0 && text_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_5.tStart = t;  // (not accounting for frame time here)
      text_5.frameNStart = frameN;  // exact frame index
      
      text_5.setAutoDraw(true);
    }

    
    // *key_resp_4* updates
    if (frameN >= 0.0 && key_resp_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_4.tStart = t;  // (not accounting for frame time here)
      key_resp_4.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_4.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.clearEvents(); });
    }

    if (key_resp_4.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_4.getKeys({keyList: ['m'], waitRelease: false});
      _key_resp_4_allKeys = _key_resp_4_allKeys.concat(theseKeys);
      if (_key_resp_4_allKeys.length > 0) {
        key_resp_4.keys = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].name;  // just the last key pressed
        key_resp_4.rt = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of CheckComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function CheckRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Check'-------
    for (const thisComponent of CheckComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Check" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_8_allKeys;
var DemoComponents;
function DemoRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Demo'-------
    t = 0;
    DemoClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_8.keys = undefined;
    key_resp_8.rt = undefined;
    _key_resp_8_allKeys = [];
    // keep track of which components have finished
    DemoComponents = [];
    DemoComponents.push(text_11);
    DemoComponents.push(key_resp_8);
    
    for (const thisComponent of DemoComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function DemoRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Demo'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = DemoClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_11* updates
    if (frameN >= 0.0 && text_11.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_11.tStart = t;  // (not accounting for frame time here)
      text_11.frameNStart = frameN;  // exact frame index
      
      text_11.setAutoDraw(true);
    }

    
    // *key_resp_8* updates
    if (frameN >= 0.0 && key_resp_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_8.tStart = t;  // (not accounting for frame time here)
      key_resp_8.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_8.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.clearEvents(); });
    }

    if (key_resp_8.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_8.getKeys({keyList: [], waitRelease: false});
      _key_resp_8_allKeys = _key_resp_8_allKeys.concat(theseKeys);
      if (_key_resp_8_allKeys.length > 0) {
        key_resp_8.keys = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].name;  // just the last key pressed
        key_resp_8.rt = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of DemoComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function DemoRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Demo'-------
    for (const thisComponent of DemoComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Demo" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var demo;
var currentLoop;
function demoLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  demo = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'testTrialLoop.xlsx',
    seed: undefined, name: 'demo'
  });
  psychoJS.experiment.addLoop(demo); // add the loop to the experiment
  currentLoop = demo;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisDemo of demo) {
    const snapshot = demo.getSnapshot();
    thisScheduler.add(importConditions(snapshot));
    thisScheduler.add(exampleRoutineBegin(snapshot));
    thisScheduler.add(exampleRoutineEachFrame(snapshot));
    thisScheduler.add(exampleRoutineEnd(snapshot));
    thisScheduler.add(backtocenterRoutineBegin(snapshot));
    thisScheduler.add(backtocenterRoutineEachFrame(snapshot));
    thisScheduler.add(backtocenterRoutineEnd(snapshot));
    thisScheduler.add(endLoopIteration(thisScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function demoLoopEnd() {
  psychoJS.experiment.removeLoop(demo);

  return Scheduler.Event.NEXT;
}


var trials;
function trialsLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 2, method: TrialHandler.Method.RANDOM,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'trialLoop.xlsx',
    seed: undefined, name: 'trials'
  });
  psychoJS.experiment.addLoop(trials); // add the loop to the experiment
  currentLoop = trials;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial of trials) {
    const snapshot = trials.getSnapshot();
    thisScheduler.add(importConditions(snapshot));
    thisScheduler.add(trialRoutineBegin(snapshot));
    thisScheduler.add(trialRoutineEachFrame(snapshot));
    thisScheduler.add(trialRoutineEnd(snapshot));
    thisScheduler.add(backtocenterRoutineBegin(snapshot));
    thisScheduler.add(backtocenterRoutineEachFrame(snapshot));
    thisScheduler.add(backtocenterRoutineEnd(snapshot));
    thisScheduler.add(Pause_1RoutineBegin(snapshot));
    thisScheduler.add(Pause_1RoutineEachFrame(snapshot));
    thisScheduler.add(Pause_1RoutineEnd(snapshot));
    thisScheduler.add(endLoopIteration(thisScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trialsLoopEnd() {
  psychoJS.experiment.removeLoop(trials);

  return Scheduler.Event.NEXT;
}


var exampleComponents;
function exampleRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'example'-------
    t = 0;
    exampleClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    image_2.setPos([coex, coey]);
    image_2.setImage(imageFile);
    // setup some python lists for storing info about the mouse_2
    // current position of the mouse:
    mouse_2.x = [];
    mouse_2.y = [];
    mouse_2.leftButton = [];
    mouse_2.midButton = [];
    mouse_2.rightButton = [];
    mouse_2.time = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    exampleComponents = [];
    exampleComponents.push(image_2);
    exampleComponents.push(mouse_2);
    exampleComponents.push(image_5);
    
    for (const thisComponent of exampleComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function exampleRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'example'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = exampleClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *image_2* updates
    if (frameN >= 60 && image_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_2.tStart = t;  // (not accounting for frame time here)
      image_2.frameNStart = frameN;  // exact frame index
      
      image_2.setAutoDraw(true);
    }

    if (image_2.status === PsychoJS.Status.STARTED && frameN >= 90) {
      image_2.setAutoDraw(false);
    }
    // *mouse_2* updates
    if (frameN >= 60 && mouse_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_2.tStart = t;  // (not accounting for frame time here)
      mouse_2.frameNStart = frameN;  // exact frame index
      
      mouse_2.status = PsychoJS.Status.STARTED;
      mouse_2.mouseClock.reset();
      prevButtonState = mouse_2.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_2.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      let buttons = mouse_2.getPressed();
      if (!buttons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = buttons;
        if (buttons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          const xys = mouse_2.getPos();
          mouse_2.x.push(xys[0]);
          mouse_2.y.push(xys[1]);
          mouse_2.leftButton.push(buttons[0]);
          mouse_2.midButton.push(buttons[1]);
          mouse_2.rightButton.push(buttons[2]);
          mouse_2.time.push(mouse_2.mouseClock.getTime());
          // abort routine on response
          continueRoutine = false;
        }
      }
    }
    
    // *image_5* updates
    if (frameN >= 0.0 && image_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_5.tStart = t;  // (not accounting for frame time here)
      image_5.frameNStart = frameN;  // exact frame index
      
      image_5.setAutoDraw(true);
    }

    if (frameN < 63) {
        document.body.style.cursor='none';
    } else {
        document.body.style.cursor='auto';
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of exampleComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function exampleRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'example'-------
    for (const thisComponent of exampleComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // store data for thisExp (ExperimentHandler)
    if (mouse_2.x) {  psychoJS.experiment.addData('mouse_2.x', mouse_2.x[0])};
    if (mouse_2.y) {  psychoJS.experiment.addData('mouse_2.y', mouse_2.y[0])};
    if (mouse_2.leftButton) {  psychoJS.experiment.addData('mouse_2.leftButton', mouse_2.leftButton[0])};
    if (mouse_2.midButton) {  psychoJS.experiment.addData('mouse_2.midButton', mouse_2.midButton[0])};
    if (mouse_2.rightButton) {  psychoJS.experiment.addData('mouse_2.rightButton', mouse_2.rightButton[0])};
    if (mouse_2.time) {  psychoJS.experiment.addData('mouse_2.time', mouse_2.time[0])};
    
    // the Routine "example" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var backtocenterComponents;
function backtocenterRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'backtocenter'-------
    t = 0;
    backtocenterClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    // setup some python lists for storing info about the mouse_4
    mouse_4.clicked_name = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    backtocenterComponents = [];
    backtocenterComponents.push(image_4);
    backtocenterComponents.push(mouse_4);
    
    for (const thisComponent of backtocenterComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function backtocenterRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'backtocenter'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = backtocenterClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *image_4* updates
    if (frameN >= 0.0 && image_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_4.tStart = t;  // (not accounting for frame time here)
      image_4.frameNStart = frameN;  // exact frame index
      
      image_4.setAutoDraw(true);
    }

    // *mouse_4* updates
    if (frameN >= 0.0 && mouse_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_4.tStart = t;  // (not accounting for frame time here)
      mouse_4.frameNStart = frameN;  // exact frame index
      
      mouse_4.status = PsychoJS.Status.STARTED;
      mouse_4.mouseClock.reset();
      prevButtonState = mouse_4.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_4.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      let buttons = mouse_4.getPressed();
      if (!buttons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = buttons;
        if (buttons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          for (const obj of [image_4]) {
            if (obj.contains(mouse_4)) {
              gotValidClick = true;
              mouse_4.clicked_name.push(obj.name)
            }
          }
          if (gotValidClick === true) { // abort routine on response
            continueRoutine = false;
          }
        }
      }
    }
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of backtocenterComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function backtocenterRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'backtocenter'-------
    for (const thisComponent of backtocenterComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // store data for thisExp (ExperimentHandler)
    // the Routine "backtocenter" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_6_allKeys;
var startComponents;
function startRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'start'-------
    t = 0;
    startClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_6.keys = undefined;
    key_resp_6.rt = undefined;
    _key_resp_6_allKeys = [];
    // keep track of which components have finished
    startComponents = [];
    startComponents.push(text_8);
    startComponents.push(key_resp_6);
    
    for (const thisComponent of startComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function startRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'start'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = startClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_8* updates
    if (frameN >= 0.0 && text_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_8.tStart = t;  // (not accounting for frame time here)
      text_8.frameNStart = frameN;  // exact frame index
      
      text_8.setAutoDraw(true);
    }

    
    // *key_resp_6* updates
    if (frameN >= 0.0 && key_resp_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_6.tStart = t;  // (not accounting for frame time here)
      key_resp_6.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_6.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_6.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_6.clearEvents(); });
    }

    if (key_resp_6.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_6.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_6_allKeys = _key_resp_6_allKeys.concat(theseKeys);
      if (_key_resp_6_allKeys.length > 0) {
        key_resp_6.keys = _key_resp_6_allKeys[_key_resp_6_allKeys.length - 1].name;  // just the last key pressed
        key_resp_6.rt = _key_resp_6_allKeys[_key_resp_6_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of startComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function startRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'start'-------
    for (const thisComponent of startComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var imageIndex;
var imageName;
var trialComponents;
function trialRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'trial'-------
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    imageIndex = Math.floor(Math.random() * 4+1)+(cond-1)*4;
    imageName = imageIndex+".bmp";
    image.setPos([coex, coey]);
    image.setImage(imageName);
    // setup some python lists for storing info about the mouse
    // current position of the mouse:
    mouse.x = [];
    mouse.y = [];
    mouse.leftButton = [];
    mouse.midButton = [];
    mouse.rightButton = [];
    mouse.time = [];
    gotValidClick = false; // until a click is received
    imageIndex = Math.floor(Math.random() * 4+1)+(cond-1)*4;
    imageName = imageIndex+".bmp";
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(image);
    trialComponents.push(mouse);
    trialComponents.push(image_6);
    
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function trialRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'trial'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *image* updates
    if (frameN >= 60 && image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image.tStart = t;  // (not accounting for frame time here)
      image.frameNStart = frameN;  // exact frame index
      
      image.setAutoDraw(true);
    }

    if (image.status === PsychoJS.Status.STARTED && frameN >= 63) {
      image.setAutoDraw(false);
    }
    // *mouse* updates
    if (frameN >= 60 && mouse.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse.tStart = t;  // (not accounting for frame time here)
      mouse.frameNStart = frameN;  // exact frame index
      
      mouse.status = PsychoJS.Status.STARTED;
      mouse.mouseClock.reset();
      prevButtonState = mouse.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      let buttons = mouse.getPressed();
      if (!buttons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = buttons;
        if (buttons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          const xys = mouse.getPos();
          mouse.x.push(xys[0]);
          mouse.y.push(xys[1]);
          mouse.leftButton.push(buttons[0]);
          mouse.midButton.push(buttons[1]);
          mouse.rightButton.push(buttons[2]);
          mouse.time.push(mouse.mouseClock.getTime());
          // abort routine on response
          continueRoutine = false;
        }
      }
    }
    
    // *image_6* updates
    if (frameN >= 0.0 && image_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_6.tStart = t;  // (not accounting for frame time here)
      image_6.frameNStart = frameN;  // exact frame index
      
      image_6.setAutoDraw(true);
    }

    if (frameN < 63) {
        document.body.style.cursor='none';
    } else {
        document.body.style.cursor='auto';
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trialRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'trial'-------
    for (const thisComponent of trialComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // store data for thisExp (ExperimentHandler)
    if (mouse.x) {  psychoJS.experiment.addData('mouse.x', mouse.x[0])};
    if (mouse.y) {  psychoJS.experiment.addData('mouse.y', mouse.y[0])};
    if (mouse.leftButton) {  psychoJS.experiment.addData('mouse.leftButton', mouse.leftButton[0])};
    if (mouse.midButton) {  psychoJS.experiment.addData('mouse.midButton', mouse.midButton[0])};
    if (mouse.rightButton) {  psychoJS.experiment.addData('mouse.rightButton', mouse.rightButton[0])};
    if (mouse.time) {  psychoJS.experiment.addData('mouse.time', mouse.time[0])};
    
    window.trialCount += 1;
    psychoJS.experiment.addData('imageNum', imageIndex);
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var RestMessage;
var _key_resp_5_allKeys;
var Pause_1Components;
function Pause_1RoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'Pause_1'-------
    t = 0;
    Pause_1Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    window.runCount += 1;
    
    if (window.runCount < 8) {
        RestMessage = "You have finished Block " + window.runCount.toString() +" out of 8 \n Feel free to take a break and PLEASE DON'T PRESS THE ESCAPE KEY OR EXIT FULL SCREEN. \n\n Hit the spacebar to continue. \n\n Please keep looking at the center of your screen in advance";
    } else {
        RestMessage = "You have finished the experiment, please hit the spacebar to get further instruction";
    }
    text_6.text = RestMessage;
    key_resp_5.keys = undefined;
    key_resp_5.rt = undefined;
    _key_resp_5_allKeys = [];
    // keep track of which components have finished
    Pause_1Components = [];
    Pause_1Components.push(text_6);
    Pause_1Components.push(key_resp_5);
    
    for (const thisComponent of Pause_1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function Pause_1RoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'Pause_1'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = Pause_1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    var breakTrial = 5;
    if (window.trialCount % breakTrial != 0) {
        continueRoutine = false;
        frameN = -1
    }
    
    // *text_6* updates
    if (frameN >= 0.0 && text_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_6.tStart = t;  // (not accounting for frame time here)
      text_6.frameNStart = frameN;  // exact frame index
      
      text_6.setAutoDraw(true);
    }

    
    // *key_resp_5* updates
    if (frameN >= 0.0 && key_resp_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_5.tStart = t;  // (not accounting for frame time here)
      key_resp_5.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_5.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_5.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_5.clearEvents(); });
    }

    if (key_resp_5.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_5.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_5_allKeys = _key_resp_5_allKeys.concat(theseKeys);
      if (_key_resp_5_allKeys.length > 0) {
        key_resp_5.keys = _key_resp_5_allKeys[_key_resp_5_allKeys.length - 1].name;  // just the last key pressed
        key_resp_5.rt = _key_resp_5_allKeys[_key_resp_5_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Pause_1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Pause_1RoutineEnd(trials) {
  return function () {
    //------Ending Routine 'Pause_1'-------
    for (const thisComponent of Pause_1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "Pause_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_7_allKeys;
var EndComponents;
function EndRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'End'-------
    t = 0;
    EndClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_7.keys = undefined;
    key_resp_7.rt = undefined;
    _key_resp_7_allKeys = [];
    // keep track of which components have finished
    EndComponents = [];
    EndComponents.push(text_10);
    EndComponents.push(key_resp_7);
    
    for (const thisComponent of EndComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function EndRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'End'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = EndClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_10* updates
    if (frameN >= 0.0 && text_10.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_10.tStart = t;  // (not accounting for frame time here)
      text_10.frameNStart = frameN;  // exact frame index
      
      text_10.setAutoDraw(true);
    }

    
    // *key_resp_7* updates
    if (frameN >= 0.0 && key_resp_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_7.tStart = t;  // (not accounting for frame time here)
      key_resp_7.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_7.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.clearEvents(); });
    }

    if (key_resp_7.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_7.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_7_allKeys = _key_resp_7_allKeys.concat(theseKeys);
      if (_key_resp_7_allKeys.length > 0) {
        key_resp_7.keys = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].name;  // just the last key pressed
        key_resp_7.rt = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of EndComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function EndRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'End'-------
    for (const thisComponent of EndComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(thisScheduler, loop) {
  // ------Prepare for next entry------
  return function () {
    if (typeof loop !== 'undefined') {
      // ------Check if user ended loop early------
      if (loop.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(loop);
        }
      thisScheduler.stop();
      } else {
        const thisTrial = loop.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(loop);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(trials) {
  return function () {
    psychoJS.importAttributes(trials.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  
  
  
  
  
  
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
