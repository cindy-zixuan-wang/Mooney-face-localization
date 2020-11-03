imOri = imread("wf.png");
% isolate the face
imFace = imOri(80:382,97:348,:);
imScrb = imscramble(imFace);
% put the face back
imComb = imOri;
imComb(80:382,97:348,:) = imScrb;
gaussAperture(imComb,80,'noisePatch.png',512);