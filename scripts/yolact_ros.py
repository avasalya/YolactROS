#! /usr/bin/env python3
from __init__ import *
# conda activate yolo6d

color_cache = defaultdict(lambda: {})

class YolactROS:

    def __init__(self, net:Yolact):
        self.net = net

        self.fps = 0
        self.top_k = 5
        self.score_threshold = 0.0

        self.crop_masks = True
        self.display_masks = True
        self.display_scores = True
        self.display_bboxes = True
        self.display_text = True
        self.display_fps = False

        # subscribe to RGB topic
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.rgb_sub.registerCallback(self.callback)

    def callback(self, Image):
        self.img = np.frombuffer(Image.data, dtype=np.uint8).reshape(Image.height, Image.width, -1)

        # segment object
        try:
            self.evalimage()
            # cv2.imshow('yolact', cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), cv2.waitKey(1)
        except rospy.ROSException:
            print(f'{Fore.RED}ROS Interrupted{Style.RESET_ALL}')


    def evalimage(self):
        with torch.no_grad():
            frame = torch.from_numpy(self.img).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)

            h, w, _ = frame.shape
            classes, scores, boxes, masks = self.postprocess_results(preds, w, h)
            print(f'{Fore.RED}Total onigiri found{Style.RESET_ALL}', len(boxes))

            image = self.prep_display(classes, scores, boxes, masks, frame, fps_str=str(self.fps))
            cv2.imshow('yolact boxes', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print('stopping, keyboard interrupt')
                # sys.exit()
                try:
                    sys.exit(1)
                except SystemExit:
                    os._exit(0)

    def prep_display(self, classes, scores, boxes, masks, img, class_color=False, mask_alpha=0.45, fps_str=''):

        img_gpu = img / 255.0

        num_dets_to_consider = min(self.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        if self.display_fps:
                # Draw the box for the fps on the GPU
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            font_thickness = 1

            text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

            img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if self.display_fps:
            # Draw the text on the CPU
            text_pt = (4, text_h + 2)
            text_color = [255, 255, 255]

            cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        if num_dets_to_consider == 0:
            return img_numpy

        if self.display_text or self.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]
                print(f'{Fore.GREEN}at confidence{Style.RESET_ALL}', score)

                if self.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if self.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if self.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_numpy


    def postprocess_results(self, dets_out, w, h):
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = self.crop_masks,
                                            score_threshold   = self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        return classes, scores, boxes, masks


if __name__ == '__main__':

    rospy.init_node('yolact_ros', anonymous=False)

    rospack       = rospkg.RosPack()
    yolact_path   = rospack.get_path('yolact_ros')
    model_path    = os.path.join(yolact_path, 'txonigiri', 'yolact_base_31999_800000.pth')
    trained_model = SavePath.from_str(model_path)
    set_cfg(trained_model.model_name + '_config')

    with torch.no_grad():
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        print('Loading model from', model_path)
        net = Yolact()
        net.load_weights(model_path)
        net.eval()
        print('Model loaded.')

        net = net.cuda()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        YolactROS(net)

    try:
        rospy.spin()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        print('Shutting down Yolact ROS node')
        cv2.destroyAllWindows()