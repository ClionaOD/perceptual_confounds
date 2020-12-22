function energy_output = apply_csf(video_name)
%Apply csf to videos. 

v = VideoReader(video_name); 
%Stats on video
Nx = v.Width;
Ny = v.Height;
%N_channels = size(frame, 3);
Nt = v.NumFrames; 


%********************************
%% Save frames of video
frames = zeros(Ny, Nx, Nt); 
%fftn_frames = zeros(Ny, Nx, N_channels, Nt); 
count = 1;

while hasFrame(v)
    frame = readFrame(v);
    frame_grey = rgb2gray(frame);
    %Determine mean 
    %meanGrayLevel = mean2(frame_grey); % This is a double.
    %Image_no_dc = double(frame_grey) - meanGrayLevel;
    frames(:,:, count) = frame_grey;
    count = count + 1;
end

%% *****************
%fft
% f=fftn(frames);
% f2 = fftshift(f);
% plot(abs(mean(mean(f2,2),3)));
% ylim([0, 0.25])
% 
% plot(abs(mean(mean(f,2),3)));
% ylim([0, 0.25])

%********************************
%% CSF
%Apply function
% degrees of visual angle
width_degrees=90;
%width_pixels=v.Width; % if video is full screen, this many pixels across 90 degrees
width_pixels=1929; % if projector has this many pixels across and video is video width (i.e., 640) pixels across
fsx = width_pixels/width_degrees;  % pixels/degree; video width in pix/ 90 degrees of visual angle
fsy = fsx; % Assume square pixels
fst = v.FrameRate %Frames/sec 

% Nx = 640 %Number of columns in each frame
% Ny = 360 %Number of rows in each frame
% Nt = 563 %Number of frames 
estab = 0; % eye stabliation
fy0 = 0;

chunk_dur_secs=2;
chunk_shift_secs=1;

chunk_dur_nframes=fst*chunk_dur_secs;
chunk_shift_nframes=fst*chunk_shift_secs;

[CSFet,csf_fx_ft,fxx,ftt]=spatio_temp_csf(fsx,fsy,fst,Nx,Ny,chunk_dur_nframes,estab,fy0);

%Apply then to now
csf_3d = then2now(CSFet,Nx);

%********************************
%% Overlap & add strategy 

count = 1;
energy_output = [];
frame_start = 1;

frame_end = chunk_dur_nframes; 
while frame_end < Nt
    %Power of fftn of video across 2 seconds worth of frames x contrast
    %sensitivity function
    chunk=frames(:,:, frame_start: frame_end);
    meanX= mean(chunk(:));
    chunk=chunk-meanX;
    energy_out_temp = abs(fftshift(fftn(chunk))).*csf_3d;
    energy_output(count)= mean(mean(mean(energy_out_temp, 1), 2), 3); 
    %Params
    frame_start = frame_start+ chunk_shift_nframes; %
    count = count + 1;
    frame_end = frame_end + chunk_shift_nframes %Update count end  
    
end


end