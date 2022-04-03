% 1 - flipping vertically
% we used matrix manipulation to do this 
original_image = imread('gumball.0.jpg');
vertically = original_image(end:-1:1,:,:); 
figure,imshow(vertically);

% 2 - flipping horizontally
% we used matrix manipulation to do this 
horizontally = original_image(:,end:-1:1,:); 
figure,imshow(horizontally);

% 6 - apply negative transformation
% we substracted our image from 255 to find the negative one
negativeImage = 255 - original_image;
figure,imshow(negativeImage)
