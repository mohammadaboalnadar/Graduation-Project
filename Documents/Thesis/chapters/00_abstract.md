# Abstract

Legged robots offer significant advantages over wheeled or tracked platforms in traversing complex, unstructured terrains. However, achieving robust and stable legged locomotion remains a major challenge due to the high-dimensional, non-linear, and underactuated dynamics of legged systems. Deep Reinforcement Learning (DRL) has emerged as a promising alternative to classical model-based control, enabling the synthesis of dynamic gaits through trial-and-error interaction in simulation. 

This thesis presents the development of a quadrupedal locomotion policy for the Unitree A1 robot, trained from scratch using Deep Reinforcement Learning. We deploy the robot model within the MuJoCo physics engine and train a multilayer perceptron (MLP) control policy using the Proximal Policy Optimization (PPO) algorithm. Proprioceptive feedback, including base orientation, velocities, joint states, and historical action commands, is mapped directly to target joint angles. To shape the gait and prevent policy degradation, we design a multi-scale Gaussian reward function for velocity tracking combined with stability and coordination penalties (base vertical velocity, orientation, and diagonal leg symmetry). A modular curriculum learning schedule is implemented to gradually introduce stability constraints, helping the agent avoid local optima such as the "lazy agent trap" of standing still.

The trained policy is evaluated on a flat simulation surface. The results demonstrate that the PPO agent successfully learns a stable, coordinated trot-like gait that tracks forward velocity commands with a mean tracking error of **0.79 m/s** and maintains stable base roll and pitch orientation within **14.89 deg² and 8.46 deg²** variance, respectively. This work establishes a baseline for model-free continuous control on quadrupedal platforms using standard reinforcement learning algorithms without hierarchical structures or reference trajectories.

**Keywords**: Deep Reinforcement Learning, Proximal Policy Optimization (PPO), Quadrupedal Locomotion, Multilayer Perceptron (MLP), MuJoCo Simulation.

---

# ملخص الدراسة

تقدم الروبوتات ذات الأرجل مزايا كبيرة مقارنة بالمنصات ذات العجلات أو المجنزرات في عبور التضاريس المعقدة وغير المنظمة. ومع ذلك، يظل تحقيق حركة أرجل قوية ومستقرة تحديًا رئيسيًا نظرًا للديناميكيات عالية الأبعاد وغير الخطية وغير المشغلة بالكامل (underactuated) للأنظمة ذات الأرجل. وقد ظهر التعلم التعزيزي العميق (DRL) كبديل واعد للتحكم الكلاسيكي القائم على النموذج، مما يتيح توليف مشيات ديناميكية من خلال التفاعل القائم على التجربة والخطأ في المحاكاة.

تقدم هذه الأطروحة تطوير سياسة حركة رباعية الأرجل للروبوت Unitree A1، تم تدريبها من الصفر باستخدام التعلم التعزيزي العميق. نقوم بنشر نموذج الروبوت داخل محرك الفيزياء MuJoCo وتدريب سياسة تحكم تعتمد على الشبكات العصبية ذات الإدراك متعدد الطبقات (MLP) باستخدام خوارزمية تحسين السياسة القريبة (PPO). يتم ربط بيانات التغذية الراجعة للحس العميق (proprioceptive feedback) -بما في ذلك توجيه القاعدة، والسرعات، وحالات المفاصل، وأوامر الحركة السابقة- مباشرة بزوايا المفاصل المستهدفة. ولتشكيل المشية ومنع تدهور السياسة، قمنا بتصميم دالة مكافأة غاوسية متعددة المقاييس لتتبع السرعة مدمجة مع عقوبات الاستقرار والتنسيق (السرعة الرأسية للقاعدة، وتوجيه الجسم، والتماثل القطري للأرجل). تم تنفيذ جدول تعلم منهجي متدرج (curriculum learning schedule) لإدخال قيود الاستقرار تدريجيًا، مما يساعد العميل الذكي (agent) على تجنب النهايات العظمى المحلية مثل "فخ العميل الكسول" المتمثل في الوقوف دون حراك.

تم تقييم السياسة المدربة على سطح محاكاة مستوٍ. وتظهر النتائج أن خوارزمية PPO نجحت في تعلم مشية خبب (trot-like gait) مستقرة ومنسقة تتتبع أوامر السرعة الأمامية بمتوسط خطأ تتبع يبلغ **0.79 م/ث** وتحافظ على استقرار توجيه القاعدة (الالتفاف والانحدار - roll and pitch) ضمن تباين قدره **14.89 درجة² و 8.46 درجة²** على التوالي. يؤسس هذا العمل لخط أساس للتحكم المستمر الخالي من النماذج (model-free) على المنصات رباعية الأرجل باستخدام خوارزميات التعلم التعزيزي القياسية دون الحاجة إلى هياكل هرمية أو مسارات حركة مرجعية.

**الكلمات المفتاحية**: التعلم التعزيزي العميق، تحسين السياسة القريبة (PPO)، الحركة رباعية الأرجل، الإدراك متعدد الطبقات (MLP)، محاكاة MuJoCo.
