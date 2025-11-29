import cv2

from analyzeSampling import convert_point_px_to_mm
from chessboardProcess import DetectedBoard
import polynomialRegression

OUTLINE_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 0)
points = []


#Function draw a point when left mouse button is clicked
def mouse_click_action(event, x, y, flags, param):
    global points
    detected_board = param[1]
    frame = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        #if there is LIMIT_POINTS points, pop the first one and remove the circle
        if len(points) == 99:
            points.pop(0)
        point = (x, y)
        points.append(point)
        draw_point(frame, point, POINT_COLOR, detected_board)
        # save_point_to_csv(x, y)

def draw_point(frame, point_px, color, detected_board: DetectedBoard):
    cv2.circle(frame, point_px, 2, color, -1)

    # Apply the inverse perspective transform to get the position in mm
    point_mm = convert_point_px_to_mm(point_px, detected_board.grid)

    # regr_point = polynomialRegression.run_regression_polynomial(detected_board.regression_model, point_px[0], point_px[1])
    regr_point = polynomialRegression.predict_real_coordinates(detected_board.regression_model, point_px[0], point_px[1])
    cv2.putText(frame, f"mm: {point_mm[0]:.2f}, {point_mm[1]:.2f}", (point_px[0], point_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"regr: {regr_point[0]:.2f}, {regr_point[1]:.2f}", (point_px[0], point_px[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, OUTLINE_COLOR, 1)

def show_chessboard_detection(frame, detected_board):
    frame = frame

    while True:
        cv2.imshow(detected_board.board_id, frame)
        # cv2.setMouseCallback("Frame", draw_point)    

        cv2.setMouseCallback(detected_board.board_id, mouse_click_action, param=[frame, detected_board])

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(detected_board.board_id, cv2.WND_PROP_VISIBLE) < 1:
            break


    # cap.release()
    cv2.destroyAllWindows()