import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
# matplotlib.rc('text', usetex=True)

from rl_agents.agents.tree_search.graphics.robust import IntervalRobustPlannerGraphics


class RobustEPCGraphics(IntervalRobustPlannerGraphics):
    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        import pygame
        robust_env = agent.robustify_env()
        cls.display_uncertainty(robust_env=robust_env, plan=agent.get_plan(), surface=sim_surface)
        if agent_surface and hasattr(agent, "sub_agent"):
            true_theta = agent.env.unwrapped.dynamics.theta
            surf_size = agent_surface.get_size()
            image_str, size = cls.plot_ellipsoid(agent.ellipsoids, true_theta, figsize=(surf_size[0]/100, surf_size[1]/100))
            surf = pygame.image.fromstring(image_str, size, "RGB")
            agent_surface.blit(surf, (0, 0))

    @classmethod
    def display_uncertainty(cls, robust_env, plan, surface, trajectory=True):
        import pygame
        horizon = 3
        if plan:
            plan = plan[1:]  # First action has already been performed
        plan = plan[:horizon]
        for action in plan:
            robust_env.step(action)
        min_traj = [o[0] for o in robust_env.unwrapped.interval_trajectory]
        max_traj = [o[1] for o in robust_env.unwrapped.interval_trajectory]
        uncertainty_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
        cls.display_traj_uncertainty(min_traj, max_traj, uncertainty_surface, surface, cls.UNCERTAINTY_TIME_COLORMAP)
        surface.blit(uncertainty_surface, (0, 0))

    @classmethod
    def plot_ellipsoid(cls, ellipsoids, true_theta, title="", figsize=(8, 6)):
        """
            Plot the hull of all Qc, Qr points for different (action, budget).

            If a threshold beta and corresponding mixture is provided, plot them.
        :param SummaryWriter writer: will log the image to tensorboard if not None
        :param epoch: timestep for tensorboard log
        :param title: figure title
        :param figsize: figure size, inches
        :return: the string description of the image, and its size
        """
        # Figure creation
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(title)
        for ellipsoid in ellipsoids[::12]:
            confidence_ellipse(ellipsoid, ax, facecolor=(1, 0.3, 0.3, 0.1),
                               edgecolor="black", linewidth=0.5, label=None)
        confidence_ellipse(ellipsoids[-1], ax, edgecolor='red', label=r"$\mathcal{C}_{[N],\delta}$")
        plt.plot(true_theta[0], true_theta[1], '.', label=r"$\theta$")
        plt.legend()
        ax.set_xlim(-0.2, 0.7)
        ax.set_ylim(-0.2, 0.7)

        # Figure export
        fig.canvas.draw()
        data_str = fig.canvas.tostring_rgb()
        plt.close()
        return data_str, fig.canvas.get_width_height()


def confidence_ellipse(ellipsoid, ax, facecolor="none", **kwargs):
    center, cov, beta = ellipsoid
    cov = np.linalg.inv(cov / beta)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipsoid = Ellipse((0, 0),
                        width=ell_radius_x * 2,
                        height=ell_radius_y * 2,
                        facecolor=facecolor,
                        **kwargs)
    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(center[0], center[1])
    ellipsoid.set_transform(transf + ax.transData)
    return ax.add_patch(ellipsoid)