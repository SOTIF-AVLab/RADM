% FORCESNLPsolver : A fast customized optimization solver.
% 
% Copyright (C) 2013-2022 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
% [OUTPUTS] = FORCESNLPsolver(INPUTS) solves an optimization problem where:
% Inputs:
% - xinit - matrix of size [5x1]
% - x0 - matrix of size [270x1]
% - all_parameters - matrix of size [3510x1]
% Outputs:
% - outputs - column vector of length 270
function [outputs] = FORCESNLPsolver(xinit, x0, all_parameters)
    
    [output, ~, ~] = FORCESNLPsolverBuildable.forcesCall(xinit, x0, all_parameters);
    outputs = coder.nullcopy(zeros(270,1));
    outputs(1:9) = output.x01;
    outputs(10:18) = output.x02;
    outputs(19:27) = output.x03;
    outputs(28:36) = output.x04;
    outputs(37:45) = output.x05;
    outputs(46:54) = output.x06;
    outputs(55:63) = output.x07;
    outputs(64:72) = output.x08;
    outputs(73:81) = output.x09;
    outputs(82:90) = output.x10;
    outputs(91:99) = output.x11;
    outputs(100:108) = output.x12;
    outputs(109:117) = output.x13;
    outputs(118:126) = output.x14;
    outputs(127:135) = output.x15;
    outputs(136:144) = output.x16;
    outputs(145:153) = output.x17;
    outputs(154:162) = output.x18;
    outputs(163:171) = output.x19;
    outputs(172:180) = output.x20;
    outputs(181:189) = output.x21;
    outputs(190:198) = output.x22;
    outputs(199:207) = output.x23;
    outputs(208:216) = output.x24;
    outputs(217:225) = output.x25;
    outputs(226:234) = output.x26;
    outputs(235:243) = output.x27;
    outputs(244:252) = output.x28;
    outputs(253:261) = output.x29;
    outputs(262:270) = output.x30;
end
