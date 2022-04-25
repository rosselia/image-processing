
% read the image
img = imread('redbird.jpg');

% display the original image
figure
imshow(img)

% convert image to YIQ format
cn = rgb2ntsc(img);

% apply histogram equalization to YIQ
cn(:,:,1)=histeq(cn(:,:,1));

% convert image to RGB format
c2 = ntsc2rgb(cn);

% display final image
figure
imshow(c2);