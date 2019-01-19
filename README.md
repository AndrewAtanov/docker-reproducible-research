# Homework assignment for ISP course Docker for reproducible research

# How to
* Build image.
  
  `build -t <image_name> .`
  `docker run -v "<path-to-save-results>:/example/results" <image_name>`
 
 * Download image from Docker hub.
    
   `docker pull andrewatanov/docker-sip`
  
   `docker run -v "<path-to-save-results>:/example/results" andrewatanov/docker-sip`
