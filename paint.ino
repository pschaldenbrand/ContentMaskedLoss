/*
  https://docs.google.com/presentation/d/1iT1aNLbc_xyVv69dLwT8w649elWUR_tekqOBhJPvgYA/edit?usp=sharing
*/

#include <math.h>
#include <Braccio.h>
#include <Servo.h>
#include <stdlib.h>

#define PI 3.14159265

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_rot;
Servo wrist_ver;
Servo gripper;


// Data transfer junk
const int numChars = 300;
char receivedChars[numChars];
boolean newData = false;

double deg2rad(double deg) { return deg * PI / 180; }
double rad2deg(double rad) { return rad * 180 / PI; }

const double beta_0_deg = 90.0;//148.1;         // Wrist angle
const double beta_0_rad = deg2rad(beta_0_deg);
const double beta_0_bracc = beta_0_deg - 90.0;

double beta_0_deg_far = 110.0;//148.1;         // Wrist angle when D is large
double beta_0_rad_far = deg2rad(beta_0_deg_far);
double beta_0_bracc_far = beta_0_deg_far - 90.0;

const double beta_1_deg = 180.0;                // The brush angle
const double beta_1_rad = deg2rad(beta_1_deg);

const double beta_2_deg = 141.47;
const double beta_2_rad = deg2rad(beta_2_deg);

const double wrist = beta_0_bracc;
const double wrist_far = beta_0_bracc_far;

const double A_0 = 12.2;
const double A_1 = 12.5;
const double A_2 = 12.5;
const double A_3 = 19.5;
const double A_4 = 4.0; // Brush length

double T = sqrt(pow(A_2,2.0) + pow(A_3,2.0) - (2.0 * A_2 * A_3 * cos(beta_0_rad))); // convert to radians?
double a = acos((pow(A_3,2.0) + pow(T,2.0) - pow(A_2,2.0)) / (2.0 * A_3 * T));
double b = beta_1_rad - a;
double C = sqrt(pow(T,2.0) + pow(A_4,2.0) - (2.0 * T * A_4 * cos(b)));

void get_thetas(double D, double *ar) {
  double E = sqrt(pow(A_0,2.0) + pow(D,2.0) - (2.0 * A_0 * D * cos(beta_2_rad)));
  
  double lambda_0 = acos((pow(A_4,2.0) + pow(C,2.0) - pow(T,2.0)) / (2.0 * A_4 * C));
  double lambda_1 = acos((pow(C,2.0) + pow(E,2.0) - pow(A_1,2.0)) / (2.0 * C * E));
  double lambda_2 = acos((pow(E,2.0) + pow(D,2.0) - pow(A_0,2.0)) / (2.0 * E * D));
  
  double alpha = lambda_0 + lambda_1 + lambda_2;
  
  double F = sqrt(pow(D,2.0) + pow(C,2.0) - (2.0 * D * C * cos(lambda_1 + lambda_2)));
  
  double theta_0_rad = acos((pow(A_0,2.0) + pow(A_1,2.0) - pow(F,2.0)) / (2.0 * A_0 * A_1));
  double theta_1_rad = deg2rad(720.0) - (beta_0_rad + beta_1_rad + beta_2_rad + theta_0_rad + alpha);
  
  // Serial.println("T = " + String(T) + "\na = " + String(a) + "\nb = " + String(b));
  // Serial.println("C = " + String(C) + "\nE = " + String(E));
  // Serial.println("lambda_0 = " + String(lambda_0) + "\nlambda_1 = " + String(lambda_1) + "\nlambda_2 = " + String(lambda_2));
  // Serial.println("alpha = " + String(alpha) + "\nF = " + String(F) + "\nb = " + String(b));
  // Serial.println("theta_0_rad = " + String(theta_0_rad) + "\ntheta_1_rad = " + String(theta_1_rad));
  // Serial.println(String(rad2deg(theta_0_rad)) + ", " + String(rad2deg(theta_1_rad)));
  ar[0] = rad2deg(theta_0_rad);
  ar[1] = rad2deg(theta_1_rad);
}

void get_thetas_far(double D, double *ar) {
  double T_far = sqrt(pow(A_2,2.0) + pow(A_3,2.0) - (2.0 * A_2 * A_3 * cos(beta_0_rad_far))); // convert to radians?
  double a_far = acos((pow(A_3,2.0) + pow(T_far,2.0) - pow(A_2,2.0)) / (2.0 * A_3 * T_far));
  double b_far = beta_1_rad - a_far;
  double C_far = sqrt(pow(T_far,2.0) + pow(A_4,2.0) - (2.0 * T_far * A_4 * cos(b_far)));

  double E = sqrt(pow(A_0,2.0) + pow(D,2.0) - (2.0 * A_0 * D * cos(beta_2_rad)));
  
  double lambda_0 = acos((pow(A_4,2.0) + pow(C_far,2.0) - pow(T_far,2.0)) / (2.0 * A_4 * C_far));
  double lambda_1 = acos((pow(C_far,2.0) + pow(E,2.0) - pow(A_1,2.0)) / (2.0 * C_far * E));
  double lambda_2 = acos((pow(E,2.0) + pow(D,2.0) - pow(A_0,2.0)) / (2.0 * E * D));
  
  double alpha = lambda_0 + lambda_1 + lambda_2;
  
  double F = sqrt(pow(D,2.0) + pow(C_far,2.0) - (2.0 * D * C_far * cos(lambda_1 + lambda_2)));
  
  double theta_0_rad = acos((pow(A_0,2.0) + pow(A_1,2.0) - pow(F,2.0)) / (2.0 * A_0 * A_1));
  double theta_1_rad = deg2rad(720.0) - (beta_0_rad_far + beta_1_rad + beta_2_rad + theta_0_rad + alpha);
  
  // Serial.println("T = " + String(T) + "\na = " + String(a) + "\nb = " + String(b));
  // Serial.println("C = " + String(C) + "\nE = " + String(E));
  // Serial.println("lambda_0 = " + String(lambda_0) + "\nlambda_1 = " + String(lambda_1) + "\nlambda_2 = " + String(lambda_2));
  // Serial.println("alpha = " + String(alpha) + "\nF = " + String(F) + "\nb = " + String(b));
  // Serial.println("theta_0_rad = " + String(theta_0_rad) + "\ntheta_1_rad = " + String(theta_1_rad));
  // Serial.println(String(rad2deg(theta_0_rad)) + ", " + String(rad2deg(theta_1_rad)));
  ar[0] = rad2deg(theta_0_rad);
  ar[1] = rad2deg(theta_1_rad);
}

double shoulder_adj = 2.0;

double deg2braccioShoulder(double theta_0_deg) {
  double shoulder = theta_0_deg - (180.0 - beta_2_deg);
  if (shoulder > 85) { shoulder = shoulder + 3.0; }
  
  shoulder = shoulder + shoulder_adj;
  if (shoulder > 180) { shoulder = 180; }
  if (shoulder < 0) {shoulder = 0; }
  return shoulder;
}

double elbow_adj = -2.0;

double deg2braccioElbow(double theta_1_deg) {
  double elbow = theta_1_deg - 90.0;

  elbow = elbow + elbow_adj;
  if (elbow > 180) { elbow = 180; }
  if (elbow < 0) {elbow = 0; }
  return elbow;
}

const double x_length = 20.0;
const double y_length = 20.0;
const double canvas_x_offset = 10.0; // Canvas doesn't have to be semetrical over median
const double buff = 10.0; // Buffer between wood and start of painting
const double bracc_base_width = 9.5; // Distance between center of braccio and edge of wood
//const double radius_adj = 12.0; // The base radius is off by this much (should be 12, but seems to work better with 0)

double adjust_radius(double radius) {
  radius = radius + 12.0;
  
  // If radius is 20 adjust -6, 80 adjust 0.0, fill in between.
  if (radius >= 20 && radius <= 80) {
    radius = radius - (1 - ((radius-20)/60)) * 6;
  }
  return radius;
}

double wrist_adj = 3.0;
double D_thresh = 18.0;

void move_on_canvas(double x, double y) {
  // double D_and_init = sqrt(pow(x-(x_length/2 + canvas_x_offset), 2.0) + pow(y + buff + bracc_base_width, 2.0));
  
  // double theta = asin((x-(x_length/2 + canvas_x_offset)) / D_and_init);
  // double radius = rad2deg(deg2rad(90.0) + theta);
  // double D = D_and_init - bracc_base_width;
  // if (D < 2) { return; }
  double L = sqrt(pow(x-canvas_x_offset, 2.0) + pow(y + buff + bracc_base_width, 2.0));
  
  double theta = asin((x-canvas_x_offset) / L);
  double radius = rad2deg(deg2rad(90.0) + theta);
  double D = L - bracc_base_width;
  
  double* thetas = new double[2];
  if (D < D_thresh) {
    get_thetas(D, thetas);
  } else {
    // Adjust Wrist for long D's
    beta_0_deg_far = 90 + (D-18) / 12 * 30; // + 0 if D = 18  + 30 if D = 30
    beta_0_rad_far = deg2rad(beta_0_deg_far);
    beta_0_bracc_far = beta_0_deg_far - 90.0;

    get_thetas_far(D, thetas);
  }
  double shoulder = deg2braccioShoulder(thetas[0]);
  double elbow = deg2braccioElbow(thetas[1]);
  
  // Serial.println("\nD_and_init = " + String(D_and_init) + "\ntheta = " + String(rad2deg(theta)) + "\nradius = " + String(radius));
  // Serial.println("init = " + String(init) + "\nD = " + String(D) + "\n");
  // Serial.print("D = ");
  // Serial.println(D);
  // Serial.print("shoulder = ");
  // Serial.println(shoulder);
  // Serial.print("elbow = ");
  // Serial.println(elbow);
  if (shoulder + elbow > 100) { // safety
    // Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, wrist+wrist_adj, 90,  73);
    if (D < D_thresh) {
      Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, wrist+wrist_adj, 90,  73);
    } else {
      Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, beta_0_bracc_far+wrist_adj, 90,  73);
    }
  } else {
    Serial.println("Safety");
  }
  free(thetas);
}

void hover_above_spot(double x, double y) {
  double L = sqrt(pow(x-canvas_x_offset, 2.0) + pow(y + buff + bracc_base_width, 2.0));
  
  double theta = asin((x-canvas_x_offset) / L);
  double radius = rad2deg(deg2rad(90.0) + theta);
  double D = L - bracc_base_width;
  
  double* thetas = new double[2];
  if (D < D_thresh) {
    get_thetas(D, thetas);
  } else {
    // Adjust Wrist for long D's
    beta_0_deg_far = 90 + (D-18) / 12 * 30; // + 0 if D = 18  + 30 if D = 30
    beta_0_rad_far = deg2rad(beta_0_deg_far);
    beta_0_bracc_far = beta_0_deg_far - 90.0;

    get_thetas_far(D, thetas);
  }
  
  double shoulder = deg2braccioShoulder(thetas[0]) + 20; // This extra degrees is what makes it hover
  double elbow = deg2braccioElbow(thetas[1]);
  
  // Serial.println("\nD_and_init = " + String(D_and_init) + "\ntheta = " + String(rad2deg(theta)) + "\nradius = " + String(radius));
  // Serial.println("init = " + String(init) + "\nD = " + String(D) + "\n");
  if (shoulder + elbow > 120) { // safety
    // Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, wrist+wrist_adj, 90,  73);
    if (D < D_thresh) {
      Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, wrist+wrist_adj, 90,  73);
    } else {
      Braccio.ServoMovement(30, adjust_radius(radius), shoulder, elbow, wrist_far+wrist_adj, 90,  73);
    }
  }
  free(thetas);
}

const double n_steps = 100;

void paint(double x0, double y0, double x1, double y1, double x2, double y2) {
  // Drag the brush between three points
  hover_above_spot(x0, y0);
  delay(300);
  double x; double y; double t;
  
  double L0 = pow(pow(x1 - x0, 2) + pow(y1-y0,2), 0.5);
  double L1 = pow(pow(x2 - x1, 2) + pow(y2-y1,2), 0.5);
  double L = L0 + L1;
  double max_L = 99999;//0.5 * x_length;
  
  double n_circles = n_steps;
  if (L <= max_L) {
    n_circles = n_steps;
  } else {
    n_circles = n_steps * (max_L / L);
  }
  
  for (double i = 0; i <= n_circles; i++) {
    t = i / n_steps;
    // x = ((1-t) * (1-t) * x0) + (2 * t * (1-t) * x1) + (t * t * x2);
    // y = ((1-t) * (1-t) * y0) + (2 * t * (1-t) * y1) + (t * t * y2);
    x = (1-t) * x0 + t * x2;
    y = (1-t) * y0 + t * y2;
    move_on_canvas(x, y);
    delay(10);
  }
  // move_on_canvas(x2, y2);
  delay(100);
  hover_above_spot(x, y);
}

const double paint_x = x_length + 4;
const double paint_y_start = 0;
const double paint_y_diff = 4.5; // Paints are this far apart in y axis
const double paint_x_diff = 4.5; // Second column of paints
const double num_paints = 6; // Number of different colored paints
const double paint_tray_length = 3; // Num of paints in vertical column

void get_paint(double paint_ind) {
  // safe_position();
  double x = paint_x + paint_x_diff * floor(paint_ind / paint_tray_length); // TODO support 2 trays of paint
  double y = paint_y_start + fmod(paint_ind, paint_tray_length) * paint_y_diff;
  
  hover_above_spot(x, y);
  delay(100);
  move_on_canvas(x, y); // Dunk the brush
  delay(100);
  move_on_canvas(x+.01, y+.01);
  delay(500);
  move_on_canvas(x, y);
  hover_above_spot(x, y);
  delay(100);
  // safe_position();
}

const double water_x = -10;
const double water_y = 5;

void dunk_in_water() {
  hover_above_spot(water_x, water_y);
  delay(10);
  move_on_canvas(water_x, water_y);
  delay(100);
  move_on_canvas(water_x, water_y + 2);
  delay(100);
  move_on_canvas(water_x, water_y);
  delay(100);
  move_on_canvas(water_x + 2, water_y);
  delay(100);
  move_on_canvas(water_x, water_y);
  move_on_canvas(water_x, water_y + 2);
  delay(100);
  move_on_canvas(water_x, water_y);
  delay(100);
  move_on_canvas(water_x + 2, water_y);
  delay(100);
  move_on_canvas(water_x, water_y);
  
  // Get that big drippp
  hover_above_spot(water_x, water_y);
  hover_above_spot(water_x+2, water_y+2);
  hover_above_spot(water_x, water_y);
  hover_above_spot(water_x+2, water_y+2);
  hover_above_spot(water_x, water_y);
  delay(100); 
}

const double sponge_x = -10;
const double sponge_y = -8;

void wipe_on_sponge() {
  hover_above_spot(sponge_x, sponge_y);
  delay(10);
  move_on_canvas(sponge_x, sponge_y);
  delay(200);
  move_on_canvas(sponge_x, sponge_y - (rand() % 4) + 3);
  delay(200);
  move_on_canvas(sponge_x + (rand() % 4) + 3, sponge_y- (rand() % 4) + 3);
  delay(200);
  move_on_canvas(sponge_x + (rand() % 4) + 3, sponge_y);
  delay(200);
  move_on_canvas(sponge_x, sponge_y);
  delay(200);
  move_on_canvas(sponge_x, sponge_y - (rand() % 4) + 2);
  delay(200);
  move_on_canvas(sponge_x + (rand() % 4) + 3, sponge_y- (rand() % 4) + 3);
  delay(200);
  move_on_canvas(sponge_x + (rand() % 4) + 2, sponge_y);
  delay(200);
  move_on_canvas(sponge_x, sponge_y);
  hover_above_spot(sponge_x, sponge_y);
  delay(100);
}

void clean_off_brush() {
  // Dunk in water and dry off the brush
  dunk_in_water();
  delay(500);
  wipe_on_sponge();
}

void safe_position() {
  // Safe position that is hovering above the canvas
  Braccio.ServoMovement(30, adjust_radius(90), 90, 90, wrist+wrist_adj, 90,  73);
}

void setup() {
  //Initialization functions and set up the initial position for Braccio
  //All the servo motors will be positioned in the "safety" position:
  //Base (M1):90 degrees
  //Shoulder (M2): 45 degrees
  //Elbow (M3): 180 degrees
  //Wrist vertical (M4): 180 degrees
  //Wrist rotation (M5): 90 degrees
  //gripper (M6): 10 degrees
  Braccio.begin();
  Serial.begin(9600);
  
  Braccio.ServoMovement(2000,         90, 90, 90, 90, 90,  73);
  // Serial.println("READY TO PLUG");
  // delay(10000);
  
  // tell the PC we are ready
  Serial.println("<Arduino is ready>");
}

void loop() {
  /*
   Step Delay: a milliseconds delay between the movement of each servo.  Allowed values from 10 to 30 msec.
   M1=base degrees. Allowed values from 0 to 180 degrees
   M2=shoulder degrees. Allowed values from 15 to 165 degrees
   M3=elbow degrees. Allowed values from 0 to 180 degrees
   M4=wrist vertical degrees. Allowed values from 0 to 180 degrees
   M5=wrist rotation degrees. Allowed values from 0 to 180 degrees
   M6=gripper degrees. Allowed values from 10 to 73 degrees. 10: the toungue is open, 73: the gripper is closed.
  */
  // calibrate();
  // paint_test();
  // dab_test();
  // calibrate();
  recvWithStartEndMarkers();
  if (newData == true) {
    Serial.print(F("<"));//Serial.print(receivedChars);//Serial.print(">");
    parse_data_and_paint();
    newData = false;
  }
}

void calibrate() {
  // get_paint(0);
  paint(0.0*x_length, 0.0*y_length, 0.0*x_length, 0.5*y_length, 0.0*x_length, 1.0*y_length);
  delay(10);
  // get_paint(0);
  paint(0.0*x_length, 1.0*y_length, 0.5*x_length, 1.0*y_length, 1.0*x_length, 1.0*y_length);
  delay(10);
  // get_paint(0);
  paint(1.0*x_length, 1.0*y_length, 1.0*x_length, 0.5*y_length, 1.0*x_length, 0.0*y_length);
  delay(10);
  // get_paint(0);
  paint(1.0*x_length, 0.0*y_length, 0.5*x_length, 0.0*y_length, 0.0*x_length, 0.0*y_length);
  delay(10);
  
  // for (double i = 0; i <= 15; i = i + 3) {
  //   for (double j = 0; j <= 30; j = j + 5) {
  //     hover_above_spot(j, i);
  //     delay(100);
  //     move_on_canvas(j, i);
  //     delay(100);
  //     move_on_canvas(j, i);
      
  //     hover_above_spot(j, i);
  //     delay(100);
  //   }
  // }
  
  // paint(0, 0, 15, 0, 30, 0);
  // delay(500);
  // paint(0, 12, 15, 12, 30, 12);
  // delay(500);
  // paint(0, 0, 0, 6, 0, 12);
  // delay(500);
  // paint(30, 0, 30, 6, 30, 12);
  // safe_position();
  
  // get_paint(0);
  // get_paint(2);
  // get_paint(3);
  // get_paint(5);
  
  // for (double color_ind = 0.0; color_ind < 6.0; color_ind++) {
  //   // clean_off_brush();
  //   get_paint(color_ind);
  //   // paint(0.0*x_length, 0.0*y_length, 0.5*x_length, 0.5*y_length, 1.0*x_length, 1.0*y_length);
  // }
  // clean_off_brush();
  // safe_position();
}

/*
    Data Transfer stuff
*/
  
void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static int ndx = 0;
  byte startMarker = 0x3C;
  byte endMarker = 0x3E;
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();
    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      }
      else {
        receivedChars[ndx] = '\0'; // terminate the string
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    }

    else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

const char comma[2] = ",";
double prev_color_ind = -1;

void parse_data_and_paint() {
  boolean first_time = true;
  // split the data into its parts
  char * strtokIndx; // this is used by strtok() as an index

  strtokIndx = strtok(receivedChars, comma);double x0 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double y0 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double x1 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double y1 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double x2 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double y2 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double w0 = atof(strtokIndx); // Don't care about thickness and opacity
  strtokIndx = strtok(NULL, comma); double w1 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double o0 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double o1 = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double color_ind = atof(strtokIndx);
  strtokIndx = strtok(NULL, comma); double color_ind1 = atof(strtokIndx); // Duplicates
  strtokIndx = strtok(NULL, comma); double color_ind2 = atof(strtokIndx);


  // Clean if the color changes
  if (color_ind != prev_color_ind) {
    clean_off_brush();
    safe_position();
  }
  prev_color_ind = color_ind;
  
  // Get the new paint
  get_paint(color_ind);

  // Paint
  paint(x0*x_length, y0*y_length, x1*x_length, y1*y_length, x2*x_length, y2*y_length);
  safe_position();
  
  Serial.print(F(" x0 ")); Serial.print(x0*x_length);
  Serial.print(F(" y0 ")); Serial.print(y0*y_length);
  Serial.print(F(" x1 ")); Serial.print(x1*x_length);
  Serial.print(F(" y1 ")); Serial.print(y1*y_length);
  Serial.print(F(" x2 ")); Serial.print(x2*x_length);
  Serial.print(F(" y2 ")); Serial.print(y2*y_length);
  Serial.print(F(" color_ind ")); Serial.print(color_ind);
  Serial.println(F(">"));
  
  // free(strtokIndx); // How the fuck does uncommenting this free break the thing??? DO NOT UNCOMMENT
}


void paint_test() {
  double color_ind = 0;
  // horizontal strokes, 5cm every 5 cm vertically
  for (double x = 0; x <= x_length - 5; x = x + 5) {
    double x_start = x;
    double x_end = x + 5;
    double x_mid = (x_start + x_end) / 2;
    for (double y = 0; y <= y_length; y = y + 5) {
      // clean_off_brush();
      // get_paint(color_ind);
      paint(x_start, y, x_mid, y, x_end, y);
      // safe_position();
    }
  }
  // vertical strokes,5cm every 5 cm horizontally
  for (double y = 0; y <= y_length - 5; y = y + 5) {
    double y_start = y;
    double y_end = y + 5;
    double y_mid = (y_start + y_end) / 2;
    for (double x = 0; x <= x_length; x = x + 5) {
      // clean_off_brush();
      // get_paint(color_ind);
      paint(x, y_start, x, y_mid, x, y_end);
      // safe_position();
    }
  }
}

void dab_test() {
  double color_ind = 0;
  
  // horizontal strokes, max length (1/3 canvas) every 3 cm vertically
  for (double x = 0; x <= x_length; x = x + 3) {
    for (double y = 0; y <= y_length; y = y + 3) {
      // get_paint(color_ind);
      hover_above_spot(x,y);
      move_on_canvas(x, y);
      hover_above_spot(x,y);
    }
  }
  
}
