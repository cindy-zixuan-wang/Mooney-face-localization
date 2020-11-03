function gaussAperture(image,gaussSD,outputName,outputSize)
    if nargin < 4
        outputSize = 0;
    end
    if ischar(image)
        imNew = double(imread(image));
    else
        imNew = double(image);
    end
    if outputSize
        imNew = imresize(imNew,[outputSize, outputSize]);
    end
    xx = linspace(-size(imNew,1)/2,size(imNew,1)/2,size(imNew,1));
    [x,y] = meshgrid(xx);
    gauss = exp(-(x.^2+y.^2)./(2*gaussSD^2));
    for layer = 1:3
        imNew(:,:,layer) = 128+imNew(:,:,layer).*gauss*0.5;
    end
    imwrite(uint8(imNew),outputName,'png');
%     circMask = 1+(sqrt(x.^2+y.^2)>size(imNew,1)/3);
%     gauss = gauss.*circMask;
%     imwrite(imNew,outputName,'png','Alpha',gauss);
end